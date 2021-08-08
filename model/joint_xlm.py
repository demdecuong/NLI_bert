import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel

from .module import IntentClassifier, SlotClassifier


class JointXLMR(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst):
        super(JointXLMR, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.roberta = XLMRobertaModel(config)  # Load pretrained bert
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.use_r3f = args.use_r3f
        self.noise_type = args.noise_type
        
        if args.use_r3f:
            if self.noise_type in {"normal"}:
                self.noise_sampler = torch.distributions.normal.Normal(
                    loc=0.0, scale=args.eps
                )
            elif self.noise_type == "uniform":
                self.noise_sampler = torch.distributions.uniform.Uniform(
                    low=-args.eps, high=args.eps
                )

            self.embedding = self.roberta.base_model.embeddings

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        total_loss = 0

        if self.use_r3f:
            token_embeddings = self.embedding(input_ids, token_type_ids=token_type_ids)
            noise = self.noise_sampler.sample(sample_shape=token_embeddings.shape).to(
                            token_embeddings)
            noised_embeddings = token_embeddings.detach().clone() + noise

            # Pass embedding representation
            noised_logits = self.roberta(inputs_embeds=noised_embeddings)[0]

            symm_kl = self._get_symm_kl(noised_logits, sequence_output)
        else:
            symm_kl = 0
        total_loss += (1 - self.args.intent_loss_coef) * symm_kl
        
        intent_logits = self.intent_classifier(pooled_output)
        if not self.args.use_attention_mask:
            tmp_attention_mask = None
        else:
            tmp_attention_mask = attention_mask

        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )
            total_loss += self.args.intent_loss_coef * intent_loss
            total_loss += intent_loss

        outputs = ((intent_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,symm_kl) + outputs
 
        return outputs  # (loss, symm_kl), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

    def _get_symm_kl(self, noised_logits, input_logits):

        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
        ) / noised_logits.size(0)