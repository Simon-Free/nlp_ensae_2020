from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

class CustomBertForClassification(BertForSequenceClassification):
    def __init__(self):
        super().__init__()
        self.hidden_features = None
        self.logit_plus_attention = None
        self.pooled = None

    def get_hidden(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        self.hidden_features = outputs
        pooled_output = outputs[1]

        self.pooled = pooled_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        self.logit_plus_attention = (logits,) + outputs[2:]    