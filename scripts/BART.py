from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer


def get_feature(batch):
    encodings = tokenizer(batch['dialogue'], text_target=batch['summary'],
                          max_length=1024, truncation=True)

    encodings = {'input_ids': encodings['input_ids'],
                 'attention_mask': encodings['attention_mask'],
                 'labels': encodings['labels']}

    return encodings


if __name__ == '__main__':
    device = 'gpu'
    model_ckpt = 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
    save_ckpt = '../models/BART'

    diasum = load_dataset('knkarthick/dialogsum')

    dialogue_len = [len(x['dialogue'].split()) for x in diasum['train']]
    summary_len = [len(x['summary'].split()) for x in diasum['train']]

    diasum_pt = diasum.map(get_feature, batched=True)
    columns = ['input_ids', 'labels', 'attention_mask']
    diasum_pt.set_format(type='torch', columns=columns)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir='bart_diasum',
        num_train_epochs=1,
        warmup_steps=500,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=1e6,
        gradient_accumulation_steps=16
    )

    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer, data_collator=data_collator,
                      train_dataset=diasum_pt['train'], eval_dataset=diasum_pt['validation'])

    trainer.train()
    trainer.save_model(save_ckpt)
