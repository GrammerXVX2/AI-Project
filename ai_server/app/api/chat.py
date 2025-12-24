import asyncio
import json
import threading
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ai_server.app.schemas import UserRequest
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


@router.post("/chat")
async def process_chat(request: UserRequest):
    user_query = request.prompt

    # 1. Session Management
    current_session_id = request.session_id or str(uuid.uuid4())
    full_history = await history_manager.aload_history(current_session_id)

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
    if request.system_prompt and request.system_prompt.strip():
        system_msg = request.system_prompt
        target_agent = CODER_MODEL if decision == "CODING" else CHATTER_MODEL
    else:
        target_agent = CHATTER_MODEL
        system_msg = "You are a helpful assistant."

        if decision == "CODING":
            target_agent = CODER_MODEL
            system_msg = "You are an expert coding assistant and knowledge base expert."

    # Fallback Logic
    agent_name = target_agent
    if not model_manager.get_model(agent_name):
        # Check loaded models
        loaded = model_manager.loaded_models.keys()
        alternatives = [k for k in loaded if k != ORCHESTRATOR_MODEL]
        if alternatives:
            agent_name = alternatives[0]
            print(f"[LOGIC] '{target_agent}' missing. Switching to '{agent_name}'.")
        else:
            return {
                "response": f"⚠️ No active models found. Please load '{CODER_MODEL}' or '{CHATTER_MODEL}'.",
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

    final_messages = (
        [{"role": "system", "content": system_msg}]
        + messages_payload
        + [{"role": "user", "content": current_content}]
    )

    # Streaming Generator
    async def generate_stream():
        model_manager.set_model_active(agent_name)
        full_response = ""

        # Cap max_tokens to model context to avoid overrun errors.
        ctx_limit = getattr(agent, "n_ctx", None)
        safe_max_tokens = request.max_tokens
        if ctx_limit:
            safe_max_tokens = min(safe_max_tokens, max(256, ctx_limit // 2))

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
            model_manager.set_all_idle()

            # Save History
            full_history.append(
                {"role": "user", "content": user_query, "timestamp": str(datetime.now())}
            )
            full_history.append(
                {"role": "assistant", "content": full_response, "timestamp": str(datetime.now())}
            )
            await history_manager.asave_history(current_session_id, full_history)

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@router.get("/sessions")
async def get_sessions():
    return await history_manager.aget_all_sessions()


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    return await history_manager.aload_history(session_id)


@router.delete("/history/{session_id}")
async def delete_history(session_id: str):
    success = await history_manager.adelete_session(session_id)
    if success:
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")
