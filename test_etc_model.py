import torch
from transformers import GPT2Tokenizer, T5ForConditionalGeneration 
tokenizer = GPT2Tokenizer.from_pretrained('RussianNLP/FRED-T5-Summarizer',eos_token='</s>')
model = T5ForConditionalGeneration.from_pretrained('RussianNLP/FRED-T5-Summarizer')
device='cuda'
model.to(device)

input_text='<LM> Сократи текст.\n Одним из самых удивительных явлений в природе является миграция монархов — бабочек, которые ежегодно преодолевают тысячи километров от Канады и США до горных лесов центральной Мексики. Этот путь занимает несколько поколений: ни одна бабочка не завершает его полностью сама. Последнее поколение, рождённое осенью, живёт до восьми месяцев — в десятки раз дольше обычного — и совершает обратный перелёт на юг. Учёные до сих пор изучают, как этим крошечным существам удаётся точно ориентироваться на таких расстояниях. Предполагается, что они используют комбинацию солнечного компаса и магнитного поля Земли. Однако утрата среды обитания, использование пестицидов и изменение климата ставят под угрозу выживание вида. Сохранение монархов требует международного сотрудничества и защиты молочая — единственного растения, на котором они откладывают яйца.'
input_ids=torch.tensor([tokenizer.encode(input_text)]).to(device)
outputs=model.generate(input_ids,eos_token_id=tokenizer.eos_token_id,
                    num_beams=5,
                    min_new_tokens=17,
                    max_new_tokens=200,
                    do_sample=True,
                    no_repeat_ngram_size=4,
                    top_p=0.9)
print(tokenizer.decode(outputs[0][1:]))