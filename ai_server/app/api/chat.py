import asyncio
import json
import traceback
import re
import threading
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ai_server.app.schemas import UpdateMessageRequest, UserRequest
from ai_server.app.config import DATA_DIR, SUMMARY_CHECKPOINT_SIZE, SUMMARY_MAX_WORDS
from ai_server.app.services.history_manager import history_manager
from ai_server.app.services.model_manager import model_manager
from ai_server.app.services.rag_service import rag_service

router = APIRouter(prefix="/api", tags=["chat"])


ORCHESTRATOR_MODEL = "Orchestrator"
CODER_MODEL = "Coder Agent"
CHATTER_MODEL = "Chatter Agent"

def _looks_like_code_heavy(text: str) -> bool:
    if not text:
        return False
    if "```" in text:
        return True
    if "Traceback (most recent call last)" in text:
        return True
    if "Exception" in text and "\n" in text:
        return True
    return False


def _looks_like_code_intent(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    cues = ["rust", "python", "js", "код", "code", "script", "скрипт", "приложение", "консоль", "napishi", "напиши", "сделай"]
    return any(cue in lower for cue in cues)


def _needs_mermaid_guardrail(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    keywords = [
        "mermaid",
        "диаграм",
        "diagram",
        "sequence diagram",
        "class diagram",
        "er diagram",
        "erd",
        "flowchart",
        "блок-схема",
        "сделай диаграмму",
    ]
    return any(k in lower for k in keywords)


def _guess_mermaid_type(text: str) -> str:
    lower = text.lower()
    if "er " in lower or "erdiagram" in lower or "entity" in lower or "erd" in lower:
        return "erDiagram"
    if "sequence" in lower or "последователь" in lower:
        return "sequenceDiagram"
    if "class" in lower or "класс" in lower:
        return "classDiagram"
    if "state" in lower or "состояни" in lower:
        return "stateDiagram"
    if "gantt" in lower or "график" in lower:
        return "gantt"
    if "journey" in lower:
        return "journey"
    if "git" in lower:
        return "gitGraph"
    return "graph TD"


async def orchestrate_route(user_query: str, context_window):
    """Lightweight orchestrator: ask small model for JSON route + confidence.

    Returns: (decision: str, confidence: float, raw: str)
    decision in {"CODING", "CHAT"}
    """

    # Minimal structural guardrails: if it's clearly code-heavy, don't waste orchestration.
    if _looks_like_code_heavy(user_query):
        return "CODING", 1.0, "heuristic:code_heavy"
    if _looks_like_code_intent(user_query):
        # Try routing, but bias toward coding if the router fails.
        intent_bias = True
    else:
        intent_bias = False

    model_manager.set_model_active(ORCHESTRATOR_MODEL)
    orc_model = model_manager.get_model(ORCHESTRATOR_MODEL)
    if not orc_model:
        return "ERROR", 0.0, "orchestrator_not_loaded"

    history_lines = []
    for msg in context_window[-5:]:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 500:
            content = content[:500] + "…"
        history_lines.append(f"{role}: {content}")

    history_block = "\n".join(history_lines) if history_lines else "(empty)"

    system_msg = (
        "You are a routing orchestrator. Output MUST be valid minified JSON only (no prose). "
        "Schema: {\"route\":\"CODING\"|\"CHAT\",\"confidence\":0..1}. "
        "If uncertain, choose CHAT with confidence <= 0.6."
    )

    # Tiny few-shot helps small routers stay in JSON mode.
    user_msg = (
        "Examples:\n"
        "User: слушай а чем отличаеться ананас от огурца\n"
        "Assistant: {\"route\":\"CHAT\",\"confidence\":0.95}\n\n"
        "User: вот traceback, помоги исправить ошибку\n"
        "Assistant: {\"route\":\"CODING\",\"confidence\":0.95}\n\n"
        f"Recent context:\n{history_block}\n\nUser message:\n{user_query}"
    )

    def _call_router(sys_text: str, usr_text: str):
        return orc_model.create_chat_completion(
            messages=[
                {"role": "system", "content": sys_text},
                {"role": "user", "content": usr_text},
            ],
            max_tokens=64,
            temperature=0.0,
            top_p=0.1,
            # Stop early if the model tries to emit markdown/code fences.
            stop=["```"],
        )

    # 1st try
    orc_res = await asyncio.to_thread(_call_router, system_msg, user_msg)

    def _extract_content(resp) -> str:
        return (
            (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content")
            or ""
        ).strip()

    raw = _extract_content(orc_res)

    def _parse_route(raw_text: str):
        candidate = raw_text
        if "{" in raw_text and "}" in raw_text:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw_text[start : end + 1]

        data = json.loads(candidate)
        route = str(data.get("route", "CHAT")).strip().upper()
        confidence = float(data.get("confidence", 0.0))
        return route, confidence

    try:
        route, confidence = _parse_route(raw)
    except Exception:
        # 2nd try (stricter): explicitly forbid markdown/code, insist on JSON.
        strict_system = (
            "Return ONLY valid minified JSON (no markdown, no code, no prose). "
            "Allowed output examples: {\"route\":\"CHAT\",\"confidence\":0.9} or {\"route\":\"CODING\",\"confidence\":0.9}."
        )
        strict_user = f"User message:\n{user_query}"
        strict_res = await asyncio.to_thread(_call_router, strict_system, strict_user)
        raw2 = _extract_content(strict_res)

        try:
            route, confidence = _parse_route(raw2)
            raw = raw2
        except Exception:
            upper = (raw2 or raw).upper()
            if "CODING" in upper and "CHAT" not in upper:
                return "CODING", 0.3, raw2 or raw
            if "CHAT" in upper and "CODING" not in upper:
                return "CHAT", 0.3, raw2 or raw
            # Default safe fallback, but honor intent if present.
            if intent_bias:
                return "CODING", 0.8, raw2 or raw
            return "CHAT", 0.0, raw2 or raw

    if route not in {"CODING", "CHAT"}:
        route = "CHAT"

    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    # Guardrail: for non-code-looking, non-intent messages, require decent confidence to route to CODING.
    if route == "CODING" and confidence < 0.7 and not _looks_like_code_heavy(user_query) and not intent_bias:
        route = "CHAT"

    return route, confidence, raw


def _log_summary_failure(session_id: str, payload: str, reason: str):
    try:
        log_path = DATA_DIR / "summary_errors.log"
        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "reason": reason,
            "payload": payload,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        traceback.print_exc()


@router.post("/chat")
async def process_chat(request: UserRequest):
    user_query = request.prompt

    # 1. Session Management
    current_session_id = request.session_id or str(uuid.uuid4())
    session_data = await history_manager.aload_session(current_session_id)
    full_history = session_data.get("messages", [])
    existing_summary = session_data.get("summary")
    existing_title = session_data.get("title")

    # Context Window
    context_window = full_history[-request.history_limit :] if full_history else []

    # 2. Orchestration
    decision, confidence, raw_decision = await orchestrate_route(user_query, context_window)

    if decision == "ERROR":
        return {
            "response": "System Error: Orchestrator not loaded.",
            "category": "ERROR",
            "rag_used": False,
            "session_id": current_session_id,
        }

    print(
        f"[ORCHESTRATOR] Raw={raw_decision} -> {decision} (conf={confidence:.2f}) | Session: {current_session_id}"
    )

    # Admin-only adjustable checkpoint size (fallback to default if no admin key).
    checkpoint_size_default = SUMMARY_CHECKPOINT_SIZE
    checkpoint_size = checkpoint_size_default
    if request.admin_key and request.summary_checkpoint_size:
        # Clamp to sane bounds to avoid excessive summary calls.
        checkpoint_size = max(3, min(200, request.summary_checkpoint_size))

    rag_context = None

    # 3. Prepare Messages
    messages_payload = []
    for msg in context_window:
        messages_payload.append({"role": msg["role"], "content": msg["content"]})

    # --- RAG ---
    # Разрешаем RAG и для CHAT-запросов, но выбор модели остаётся по decision (CHAT -> Chatter).
    if request.use_rag:
        rag_context = await rag_service.asearch(user_query)

    # Select Agent
    mermaid_hint = None
    if _needs_mermaid_guardrail(user_query):
        mermaid_type = _guess_mermaid_type(user_query)
        mermaid_hint = (
            "Если отвечаешь Mermaid-диаграммой: используй тип "
            f"{mermaid_type}. Оборачивай единственный блок в ```mermaid ...```.\n"
            "Не используй несуществующие типы (например erdiag), избегай нестандартных style.\n"
            "Поддерживаемые типы: graph, sequenceDiagram, classDiagram, stateDiagram, erDiagram, gantt, journey, gitGraph."
        )

    if request.system_prompt and request.system_prompt.strip():
        system_msg = request.system_prompt
        target_agent = CODER_MODEL if decision == "CODING" else CHATTER_MODEL
    else:
        target_agent = CHATTER_MODEL
        system_msg = "You are a helpful assistant."

        if decision == "CODING":
            target_agent = CODER_MODEL
            system_msg = "You are an expert coding assistant and knowledge base expert."

    if mermaid_hint:
        system_msg = f"{system_msg}\n\n{mermaid_hint}"

    # Fallback Logic
    agent_name = target_agent
    if not model_manager.get_model(agent_name):
        loaded = list(model_manager.loaded_models.keys())

        def pick_fallback(preferred: str, secondary: str):
            for name in (preferred, secondary):
                if name and name in loaded:
                    return name
            for name in loaded:
                if name in {ORCHESTRATOR_MODEL, "Summarizer"}:
                    continue
                return name
            return None

        fallback = pick_fallback(
            CHATTER_MODEL if decision == "CHAT" else CODER_MODEL,
            CODER_MODEL if decision == "CHAT" else CHATTER_MODEL,
        )

        if fallback:
            agent_name = fallback
            print(f"[LOGIC] '{target_agent}' missing. Switching to '{agent_name}'.")
        else:
            return {
                "response": f"⚠️ No active chat/coding models found. Please load '{CODER_MODEL}' or '{CHATTER_MODEL}'.",
                "category": decision,
                "rag_used": bool(rag_context),
                "session_id": current_session_id,
            }

    print(f"[LOGIC] Using {agent_name} for {decision} task")
    agent = model_manager.get_model(agent_name)

    if agent is None:
        raise HTTPException(status_code=500, detail=f"Model '{agent_name}' is not loaded.")

    # Context Injection
    if rag_context:
        current_content = (
            f"Context from knowledge base:\n{rag_context}\n\nUser Query:\n{user_query}"
        )
    else:
        current_content = user_query

    summary_system = []
    if existing_summary:
        summary_system.append({"role": "system", "content": f"Сжатая выжимка чата: {existing_summary}"})

    final_messages = (
        [{"role": "system", "content": system_msg}]
        + summary_system
        + messages_payload
        + [{"role": "user", "content": current_content}]
    )

    async def summarize_chat(history: list, prev_summary: Optional[str]):
        def _clean_text(text: str) -> str:
            # Drop fenced/inline code and trim noisy symbols.
            cleaned = re.sub(r"```[\s\S]*?```", " ", text)
            cleaned = re.sub(r"`[^`]*`", " ", cleaned)
            cleaned = re.sub(r"[{}<>];", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned)
            return cleaned.strip()

        def _normalize(text: str) -> str:
            base = re.sub(r"[^\w\s]", " ", text.lower())
            return re.sub(r"\s+", " ", base).strip()

        def _detect_language(text: str) -> str:
            low = text.lower()
            if "c#" in low or "csharp" in low or "using system" in low:
                return "C#"
            if "python" in low or "def " in low or "import " in low:
                return "Python"
            if "javascript" in low or "typescript" in low or "console.log" in low or "node" in low:
                return "JavaScript"
            if "java" in low and "public class" in low:
                return "Java"
            if "rust" in low or "fn main" in low:
                return "Rust"
            if "go" in low and "func main" in low:
                return "Go"
            return "неизвестно"

        def _fallback_from_history(history: list, max_words: int = SUMMARY_MAX_WORDS):
            # Heuristic fallback when the summarizer misbehaves: take recent plain text, drop code, add language hint.
            recent_texts = []
            for m in reversed(history):
                content = (m.get("content") or "").strip()
                if content:
                    recent_texts.append(content)
                if len(recent_texts) >= 2:
                    break
            merged = " ".join(recent_texts)[:1200]
            cleaned = _clean_text(merged)
            lang = _detect_language(cleaned)
            summary_body = " ".join(cleaned.split()[:max_words]) if cleaned else "Диалог продолжается."
            summary = f"Язык: {lang}. {summary_body}".strip()
            title = " ".join(summary_body.split()[:8]) or "Chat"
            return title, summary

        summarizer = model_manager.get_model("Summarizer")
        if summarizer is None:
            summarizer = model_manager.load_model("Summarizer")
        if summarizer is None:
            print("[SUMMARY] Summarizer model unavailable")
            return _fallback_from_history(history)

        # Take recent messages, drop most code/noise, and build a short context block.
        recent = history[-12:]
        lines = []
        for m in recent:
            role = m.get("role", "")
            content = (m.get("content") or "")[:500]
            if content:
                content = _clean_text(content)
                if not content:
                    continue
                lines.append(f"{role}: {content}")
        messages_text = "\n".join(lines) if lines else "(empty)"

        sys = (
            "Ты ассистент, делаешь краткий вывод по диалогу на русском языке. Верни строго JSON без пояснений: "
            "{\"title\":\"краткий заголовок <= 8 слов\",\"summary\":\"выжимка 20-30 слов, без кода и без markdown, укажи язык если понятен (например 'Язык: C#')\"}."
            " Не добавляй кавычки вне JSON, не используй markdown и не копируй текст сообщения дословно."
        )
        usr = (
            f"Предыдущая выжимка: {prev_summary or '(нет)'}\n"
            f"Сообщения:\n{messages_text}"
        )

        def _safe_title_summary(title: str, summary: str):
            t = (title or "").strip()
            s = (summary or "").strip()

            def _fallback_title():
                if s:
                    return " ".join(s.split()[:8])
                first_user = next((m.get("content", "") for m in history if m.get("role") == "user" and m.get("content")), "")
                return " ".join(first_user.split()[:8]) or "Chat"

            bad_title = not t or "краткий заголовок" in t.lower() or "title" in t.lower() or "<=" in t or "слов" in t.lower()
            if bad_title:
                t = _fallback_title()
            return t or None, s or None

        def _call():
            return summarizer.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                max_tokens=180,
                temperature=0.2,
                top_p=0.3,
                stop=["<", "```"],
            )

        try:
            res = await asyncio.to_thread(_call)
            content = (
                (res.get("choices", [{}])[0].get("message", {}) or {}).get("content")
                or ""
            )
            # try to extract JSON
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}")
                blob = content[start : end + 1]
            else:
                blob = content
            try:
                data = json.loads(blob)
                title = str(data.get("title") or "").strip()
                summary = str(data.get("summary") or "").strip()
                if not summary:
                    return _fallback_from_history(history)
                # If summarizer parrots the input, fall back.
                if summary.lower() in messages_text.lower():
                    return _fallback_from_history(history)
                # Heuristic: if normalized summary is mostly contained in normalized context, treat as parroting.
                norm_sum = _normalize(summary)
                norm_ctx = _normalize(messages_text)
                if norm_sum and norm_sum[:120] in norm_ctx:
                    return _fallback_from_history(history)
                return _safe_title_summary(title, summary)
            except Exception:
                print(f"[SUMMARY] Non-JSON summary response: {blob!r}")
                _log_summary_failure(current_session_id, blob, "non_json")
                # Try strict retry without <think>
                def _strict_call():
                    strict_sys = (
                        "Верни строго JSON без пояснений и без <think>. "
                        f"Формат: {{\"title\":\"краткий заголовок <= 8 слов\",\"summary\":\"выжимка до {SUMMARY_MAX_WORDS} слов\"}}. "
                        "Никакого текста вне JSON."
                    )
                    strict_usr = usr
                    return summarizer.create_chat_completion(
                        messages=[
                            {"role": "system", "content": strict_sys},
                            {"role": "user", "content": strict_usr},
                        ],
                        max_tokens=160,
                        temperature=0.0,
                        top_p=0.1,
                        stop=["<", "```"],
                    )

                try:
                    strict_res = await asyncio.to_thread(_strict_call)
                    strict_content = (
                        (strict_res.get("choices", [{}])[0].get("message", {}) or {}).get("content")
                        or ""
                    )
                    if "{" in strict_content and "}" in strict_content:
                        s_start = strict_content.find("{")
                        s_end = strict_content.rfind("}")
                        strict_blob = strict_content[s_start : s_end + 1]
                    else:
                        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", strict_content).strip()
                        if "{" in cleaned and "}" in cleaned:
                            s_start = cleaned.find("{")
                            s_end = cleaned.rfind("}")
                            strict_blob = cleaned[s_start : s_end + 1]
                        else:
                            strict_blob = strict_content

                    data = json.loads(strict_blob)
                    title = str(data.get("title") or "").strip()
                    summary = str(data.get("summary") or "").strip()
                    if not summary:
                        return _fallback_from_history(history)
                    if summary.lower() in messages_text.lower():
                        return _fallback_from_history(history)
                    norm_sum = _normalize(summary)
                    norm_ctx = _normalize(messages_text)
                    if norm_sum and norm_sum[:120] in norm_ctx:
                        return _fallback_from_history(history)
                    return _safe_title_summary(title, summary)
                except Exception:
                    print(f"[SUMMARY] Strict retry failed: {strict_content if 'strict_content' in locals() else ''!r}")
                    _log_summary_failure(current_session_id, strict_content if 'strict_content' in locals() else traceback.format_exc(), "strict_fail")
                    return _fallback_from_history(history)
        except Exception as e:
            print(f"[SUMMARY] Failed to summarize chat {current_session_id}: {e}")
            _log_summary_failure(current_session_id, traceback.format_exc(), "exception")
            return _fallback_from_history(history)

    # Streaming Generator
    async def generate_stream():
        model_manager.set_model_active(agent_name)
        full_response = ""

        # Cap max_tokens to model context to avoid overrun errors.
        ctx_limit = getattr(agent, "n_ctx", None)
        if callable(ctx_limit):
            ctx_limit = ctx_limit()

        safe_max_tokens = request.max_tokens
        if isinstance(ctx_limit, int):
            safe_max_tokens = min(safe_max_tokens, min(16384, ctx_limit // 2))

        yield f"data: {json.dumps({'type': 'metadata', 'category': decision, 'rag_used': bool(rag_context), 'model': agent_name, 'session_id': current_session_id})}\n\n"

        token_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def generation_worker():
            try:
                stream = agent.create_chat_completion(
                    messages=final_messages,
                    max_tokens=safe_max_tokens,
                    temperature=request.temperature,
                    repeat_penalty=request.repeat_penalty,
                    stream=True,
                )

                for chunk in stream:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            asyncio.run_coroutine_threadsafe(
                                token_queue.put(delta["content"]), loop
                            )
            except Exception as e:
                print(f"Generation Error: {e}")
            finally:
                asyncio.run_coroutine_threadsafe(token_queue.put(None), loop)

        threading.Thread(target=generation_worker, daemon=True).start()

        try:
            while True:
                token = await token_queue.get()
                if token is None:
                    break

                full_response += token
                yield f"data: {json.dumps({'type': 'content', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        finally:
            # Save History
            full_history.append(
                {"role": "user", "content": user_query, "timestamp": str(datetime.now())}
            )
            full_history.append(
                {"role": "assistant", "content": full_response, "timestamp": str(datetime.now())}
            )

            new_title, new_summary = existing_title, existing_summary
            # Правила обновления:
            # - заголовок: после первого сообщения (message_count >= 1), если его ещё нет
            # - первая выжимка: один раз после >= 3 сообщений, если ещё нет summary
            # - далее: когда число сообщений кратно history_limit
            safe_limit = max(1, request.history_limit)
            message_count = len(full_history)
            initial_title_needed = existing_title is None and message_count >= 1
            initial_summary_needed = existing_summary is None and message_count >= 3
            periodic_needed = message_count >= safe_limit and message_count % safe_limit == 0
            checkpoint_due = message_count % checkpoint_size == 0
            should_summarize = initial_title_needed or initial_summary_needed or periodic_needed or checkpoint_due

            if agent is not None and should_summarize:
                t, s = await summarize_chat(full_history, existing_summary)
                if t:
                    new_title = t
                if s:
                    new_summary = s

                # Checkpoint summary over recent window
                if checkpoint_due:
                    window = full_history[-checkpoint_size:]
                    ct, cs = await summarize_chat(window, None)
                    if cs:
                        existing_checkpoints = session_data.get("summary_checkpoints", [])
                        start_idx = max(1, message_count - checkpoint_size + 1)
                        end_idx = message_count
                        existing_checkpoints.append(
                            {
                                "range": f"{start_idx}-{end_idx}",
                                "summary": cs,
                                "created_at": datetime.now().isoformat(),
                            }
                        )
                        session_data["summary_checkpoints"] = existing_checkpoints

            await history_manager.asave_history(
                current_session_id,
                full_history,
                title=new_title,
                summary=new_summary,
                summary_checkpoints=session_data.get("summary_checkpoints", []),
            )

            model_manager.set_all_idle()

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@router.get("/sessions")
async def get_sessions():
    return await history_manager.aget_all_sessions()


@router.get("/sessions/summary")
async def get_sessions_with_summary():
    """Admin-oriented endpoint: returns sessions with summaries (no auth yet)."""
    return await history_manager.aget_all_sessions()


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    return await history_manager.aload_history(session_id)


@router.put("/history/{session_id}/message/{index}")
async def update_message(session_id: str, index: int, payload: UpdateMessageRequest):
    session = await history_manager.aload_session(session_id)
    messages = session.get("messages", [])
    if index < 0 or index >= len(messages):
        raise HTTPException(status_code=404, detail="Message not found")

    messages[index]["content"] = payload.content
    messages[index]["updated_at"] = datetime.now().isoformat()

    await history_manager.asave_history(
        session_id,
        messages,
        title=session.get("title"),
        summary=session.get("summary"),
    )

    return {"status": "updated", "session_id": session_id, "index": index}


@router.delete("/history/{session_id}")
async def delete_history(session_id: str):
    success = await history_manager.adelete_session(session_id)
    if success:
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")
    