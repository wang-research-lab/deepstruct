import string
import datetime
import random
import re
import torch
import torch.nn.functional as F
import mpu
from utils import print_rank_0
from generation_utils import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, \
    NoRepeatNGramLogitsProcessor
from rouge_score import rouge_scorer


def _is_digit(w):
    for ch in w:
        if not (ch.isdigit() or ch == ','):
            return False
    return True


gigaword_tok_dict = {"(": "-lrb-", ")": "-rrb-",
                     "[": "-lsb-", "]": "-rsb-",
                     "{": "-lcb-", "}": "-rcb-",
                     "[UNK]": "UNK", '&': '&amp;', '<': '&lt;', '>': '&gt;'}

cnndm_tok_dict = {"(": "-LRB-", ")": "-RRB-",
                  "[": "-LSB-", "]": "-RSB-",
                  "{": "-LCB-", "}": "-RCB-"}


def fix_tokenization(text, dataset):
    if dataset == 'cnn_dm_org':
        return text
    if dataset == 'gigaword':
        text = text.replace('[UNK]', 'UNK')
        return text
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and \
                input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'" + input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(
                input_tokens) - 1 and _is_digit(input_tokens[i + 1]):

            output_tokens[-1] += ',' + input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and \
                input_tokens[i + 1].isdigit():

            output_tokens[-1] += '.' + input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[
            -1].isalpha() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[
            i + 1].isalpha() and input_tokens[i + 2] == '.':

            k = i + 3
            while k + 2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isalpha() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i = k
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)


def count_tokens(tokens):
    counter = {}
    for t in tokens:
        if t in counter.keys():
            counter[t] += 1
        else:
            counter[t] = 1
    return counter


def get_f1(text_a, text_b):
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if len(tokens_a) == 0 or len(tokens_b) == 0:
        return 1 if len(tokens_a) == len(tokens_b) else 0
    set_a = count_tokens(tokens_a)
    set_b = count_tokens(tokens_b)
    match = 0
    for token in set_a.keys():
        if token in set_b.keys():
            match += min(set_a[token], set_b[token])
    p = match / len(tokens_a)
    r = match / len(tokens_b)
    return 2.0 * p * r / (p + r + 1e-5)


def remove_duplicate(l_list, duplicate_rate):
    tk_list = [l.lower().split() for l in l_list]
    r_list = []
    history_set = set()
    for i, w_list in enumerate(tk_list):
        w_set = set(w_list)
        if len(w_set & history_set) / len(w_set) <= duplicate_rate:
            r_list.append(l_list[i])
        history_set |= w_set
    return r_list


def rouge_metric(predictions, labels, examples, metric="rouge-1", duplicate_rate=0.7, dataset='cnn_dm'):

    return 0.


def process_batch(batch, args):
    """Process batch and produce inputs for the model."""
    tokens = batch['text'].long().cuda()
    attention_mask = batch['attention_mask'].long().cuda()
    position_ids = batch['position_id'].long().cuda()
    return tokens, attention_mask, position_ids


TASK_RESERVED_TOKENS = {
    'conll04': " [ ] ( ) ; instance of location other human organization",
    'conll04_re': " [ ] ( ) ; instance of lives in organization based in works for located in kills",
    'nyt': " [ ] ( ) ; instance of human location organization",
    'nyt_re': " [ ] ( ) ; neighborhood of major shareholder of contains country company place lived religion people geographic distribution nationality administrative divisions founders major shareholders location ethnicity teams capital place of death place of birth place founded advisors children",
    'ade': " [ ] ( ) ; instance of disease drug",
    'ade_re': " [ ] ( ) ; effect",
    'ade0': " [ ] ( ) ; instance of disease drug",
    'ade_re0': " [ ] ( ) ; effect",
    'ace2005_joint_er': " [ ] ( ) ; instance of location geographical entity organization weapon human facility vehicle",
    'ace2005_joint_er_re': " [ ] ( ) ; part of social employer artifact located in",

    'ace2005_ner': " [ ] ( ) ; instance of facility location human organization vehicle geographical entity weapon",
    'conll03': " [ ] ( ) ; instance of location human organization miscellaneous",
    'genia': " [ ] ( ) ; instance of protein cell line dna rna cell type",
    'ontonotes': " [ ] ( ) ; instance of human facility quantity law percent date ordinal organization product monetary language cardinal nationality religious political group country city state work_of_art event location time",
    
    'conll05_srl': " [ ] ( ) ; instance of second argument first argument modal third argument fourth argument negation",
    'conll05_srl_wsj': " [ ] ( ) ; instance of fourth argument modal negation first argument third argument second argument",
    'conll05_srl_brown': " [ ] ( ) ; instance of negation modal third argument second argument fourth argument first argument",
    'conll12_srl': " [ ] ( ) ; instance of manner general - purpose general-purpose cause reference to fourth argument predication extent reference to location purpose negation reference to general-purpose sixth argument reference to second argument location reference to extent fourth argument modal direction second argument reference to fifth argument fifth argument reference to cause discourse marker reference to direction reference to third argument temporal reference to predication reference to temporal reference to manner first argument reference to first argument reciprocal third argument",
}


def create_indices(tokens, tokenizer, task_name):
    indices = []
    for batch_idx in range(tokens.shape[0]):
        indices.append(torch.LongTensor(
            sorted(tokenizer.text_tokenizer.encode(TASK_RESERVED_TOKENS[task_name]) +
                   list(set(tokens.detach().cpu().tolist()[batch_idx])) +
                   [ct.Id for ct in tokenizer._command_tokens])
        ).to(tokens.device))
    return indices


def create_mask(tokens, batch_size, vocab_size, indices):
    mask = torch.full((batch_size, vocab_size), -1e9).float().to(tokens.device)
    for idx in range(batch_size):
        mask[idx, indices[idx]] = 0
    return mask


class DecoderEvaluater:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.start_token = tokenizer.get_command('sop').Id
        self.end_token = tokenizer.get_command('eop').Id
        self.mask_token = tokenizer.get_command('sMASK').Id if args.task_mask else tokenizer.get_command('MASK').Id
        self.pad_token = tokenizer.get_command('pad').Id
        self.processors = LogitsProcessorList()
        if args.min_tgt_length > 0:
            processor = MinLengthLogitsProcessor(args.min_tgt_length, self.end_token)
            self.processors.append(processor)
        if args.no_repeat_ngram_size > 0:
            processor = NoRepeatNGramLogitsProcessor(args.no_repeat_ngram_size)
            self.processors.append(processor)

    def evaluate(self, model, dataloader, example_dict, args):
        """Calculate correct over total answers and return prediction if the
        `output_predictions` is true."""
        model.eval()
        local_predictions = {}
        print_rank_0("Distributed store created")
        with torch.no_grad():

            for idx, data in enumerate(dataloader):
                tokens, attention_mask, position_ids = process_batch(data, args)
                batch_size = tokens.size(0)
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=args.out_seq_length,
                    num_beams=args.num_beams,
                    device=tokens.device,
                    length_penalty=args.length_penalty,
                    do_early_stopping=False,
                )
                beam_scores = torch.zeros((batch_size, args.num_beams), dtype=torch.float, device=tokens.device)
                beam_scores[:, 1:] = -1e9
                beam_scores = beam_scores.view((batch_size * args.num_beams,))

                if args.data_dir.split('/')[-1] in TASK_RESERVED_TOKENS:
                    mask_indices = create_indices(tokens, self.tokenizer, args.data_dir.split('/')[-1])

                counter = 0
                while counter < args.tgt_seq_length:
                    if counter == 0:
                        next_token_logits, *mems = model(tokens, position_ids, attention_mask, return_memory=True)
                        seq_length = next_token_logits.size(1)
                        next_token_logits = next_token_logits[:, -1]
                        next_token_logits = next_token_logits.unsqueeze(1).repeat(1, args.num_beams, 1).view(
                            batch_size * args.num_beams, -1)
                        mems = [mem.unsqueeze(1).repeat(1, args.num_beams, 1, 1).view(batch_size * args.num_beams,
                                                                                      seq_length, -1) for mem in mems]
                        position_ids = tokens.new_ones(batch_size, args.num_beams, 2, 1)
                        for i, text in enumerate(tokens.tolist()):
                            mask_pos = text.index(self.mask_token)
                            position_ids[i, :, 0] = mask_pos
                        position_ids = position_ids.reshape(batch_size * args.num_beams, 2, 1)
                        tokens = tokens.new_zeros(batch_size * args.num_beams, 0)
                        attention_mask = tokens.new_zeros([batch_size * args.num_beams])
                    else:
                        if not args.no_block_position:
                            position_ids[:, 1] = counter + 1
                        last_token = tokens[:, -1:]
                        next_token_logits, *mems = model(last_token, position_ids, attention_mask, *mems,
                                                         return_memory=True)
                        next_token_logits = next_token_logits[:, -1]

                    next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                    if args.data_dir.split('/')[-1] in TASK_RESERVED_TOKENS:
                        next_token_scores += create_mask(next_token_logits, batch_size, next_token_logits.shape[-1],
                                                         mask_indices).unsqueeze(1).repeat(1, args.num_beams, 1).view(
                            batch_size * args.num_beams, -1)
                    next_token_scores = self.processors(tokens, next_token_scores)
                    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
                    vocab_size = next_token_scores.shape[-1]

                    next_token_scores = next_token_scores.view(batch_size, args.num_beams * vocab_size)

                    probs = F.softmax(next_token_scores, dim=-1)
                    if args.select_topk:
                        _, next_tokens = torch.topk(probs, k=2 * args.num_beams, dim=-1, largest=True)
                    else:
                        next_tokens = torch.multinomial(probs, num_samples=2 * args.num_beams)
                    next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                    next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                    next_tokens = torch.gather(next_tokens, -1, _indices)

                    next_indices = next_tokens // vocab_size
                    next_tokens = next_tokens % vocab_size

                    beam_outputs = beam_scorer.process(
                        tokens,
                        next_token_scores,
                        next_tokens,
                        next_indices,
                        eos_token_id=self.end_token,
                        pad_token_id=self.pad_token
                    )
                    beam_scores = beam_outputs["next_beam_scores"]
                    beam_next_tokens = beam_outputs["next_beam_tokens"]
                    beam_idx = beam_outputs["next_beam_indices"]
                    beam_next_tokens = beam_next_tokens.unsqueeze(-1)
                    tokens = torch.cat([tokens[beam_idx, :], beam_next_tokens], dim=-1)
                    mems = [mem[beam_idx] for mem in mems] if mems else []
                    if beam_scorer.is_done:
                        break
                    counter += 1
                tokens, _, scores = beam_scorer.finalize(tokens, beam_scores, next_tokens, next_indices,
                                                         eos_token_id=self.end_token, pad_token_id=self.pad_token)
                uid_list = data['uid']
                if isinstance(uid_list, torch.Tensor):
                    uid_list = uid_list.cpu().numpy().tolist()
                predictions = []
                for i, text in enumerate(tokens.tolist()):
                    text = [token for token in text if token not in [self.end_token, self.pad_token]]
                    if args.task in ['squad', 'squad_v1'] and args.tokenizer_model_type.startswith('bert'):
                        uid = uid_list[i]
                        example = example_dict[uid]
                        text = squad_decode(example, text, self.tokenizer)
                    else:
                        text = self.tokenizer.DecodeIds(text)
                    predictions.append(text)
                for uid, prediction in zip(uid_list, predictions):
                    local_predictions[uid] = prediction
                if (idx + 1) % args.log_interval == 0:
                    print_rank_0(f"Iteration {idx + 1} / {len(dataloader)}")
        model.train()
        torch.distributed.barrier()
        print_rank_0("Evaluation completed")
        gathered_predictions = [None for i in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_predictions, local_predictions)
        gathered_predictions = {uid: pred for preds in gathered_predictions for uid, pred in preds.items()}
        predictions, examples, scores = [], [], []
        for uid, example in example_dict.items():
            prediction = gathered_predictions[uid]
            predictions.append(prediction)
            examples.append(example)
        torch.distributed.barrier()
        return predictions, [], examples


def blanklm_fix_tokenization(text):
    text = text.replace("` `", "``")
    text = text.replace("\' \'", "\'\'")
    text = text.replace("n \' t", "n\'t")
    text = text.replace("\' s", "\'s")
    text = text.replace("\' m", "\'m")
    text = text.replace("\' re", "\'re")
    text = text.replace(". . .", "...")
    text = text.replace(" . .", " ..")
    text = text.replace("- -", "--")
    text = text.replace("u . s .", "u.s.")
    text = text.replace("u . k .", "u.k.")
    text = text.replace("e . g .", "e.g.")
    return text


class BlankLMEvaluater(DecoderEvaluater):

    def evaluate(self, model, dataloader, example_dict, args):
        model.eval()
        store = torch.distributed.TCPStore(args.master_ip, 18931 + random.randint(0, 10000),
                                           mpu.get_data_parallel_world_size(),
                                           torch.distributed.get_rank() == 0, datetime.timedelta(seconds=30))
        print_rank_0("Distributed store created")

        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                tokens, attention_mask, position_ids = process_batch(data, args)
                src_tokens = tokens
                batch_size = tokens.size(0)
                mask_positions = []
                current_mask = []
                for text in tokens.tolist():
                    mask_positions.append([i for i, x in enumerate(text) if x == self.mask_token])
                    current_mask.append(0)


                counter = 0
                done = [False] * batch_size
                while counter < args.tgt_seq_length:
                    if counter == 0:


                        next_token_logits, *mems = model(tokens, position_ids, attention_mask, return_memory=True)
                        next_token_logits = next_token_logits[:, -1]
                        position_ids = tokens.new_ones(batch_size, 2, 1)
                        for i, text in enumerate(tokens.tolist()):
                            mask_pos = mask_positions[i][current_mask[i]]
                            position_ids[i, 0] = mask_pos
                        tokens = tokens.new_zeros(batch_size, 0)
                        attention_mask = tokens.new_zeros(batch_size)
                    else:
                        position_ids[:, 1] = position_ids[:, 1] + 1
                        last_token = tokens[:, -1:]
                        next_token_logits, *mems = model(last_token, position_ids, attention_mask, *mems,
                                                         return_memory=True)
                        next_token_logits = next_token_logits[:, -1]
                    next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                    next_token_scores = self.processors(tokens, next_token_scores)
                    next_tokens = next_token_scores.max(dim=-1)[1]

                    for i, next_token in enumerate(next_tokens.tolist()):
                        if next_token == self.end_token:
                            if current_mask[i] + 1 < len(mask_positions[i]):
                                current_mask[i] += 1
                                next_tokens[i] = self.start_token
                                position_ids[i, 0] = mask_positions[i][current_mask[i]]
                                position_ids[i, 1] = 0
                            else:
                                done[i] = True
                        if done[i]:
                            next_tokens[i] = self.pad_token
                    if all(done):
                        break
                    tokens = torch.cat([tokens, next_tokens.unsqueeze(-1)], dim=-1)
                    counter += 1
                predictions = []
                for i, text in enumerate(tokens.tolist()):
                    text = [token for token in text if token not in [self.end_token, self.pad_token]]
                    blanks = [[]]
                    for token in text:
                        if token == self.start_token:
                            blanks.append([])
                        else:
                            blanks[-1].append(token)
                    output_tokens = []
                    current_blank = 0
                    for token in src_tokens[i].tolist():
                        if token == self.mask_token:
                            if current_blank < len(blanks):
                                output_tokens += blanks[current_blank]
                            current_blank += 1
                        else:
                            if token not in [self.pad_token]:
                                output_tokens.append(token)
                    text = self.tokenizer.DecodeIds(output_tokens[:-1])
                    text = blanklm_fix_tokenization(text)
                    predictions.append(text)

                uid_list = data['uid']
                if isinstance(uid_list, torch.Tensor):
                    uid_list = uid_list.cpu().numpy().tolist()
                for uid, prediction in zip(uid_list, predictions):
                    store.set(uid, prediction)
                if (idx + 1) % args.log_interval == 0:
                    print_rank_0(f"Iteration {idx + 1} / {len(dataloader)}")

        model.train()
        torch.distributed.barrier()
        print_rank_0("Evaluation completed")
        predictions, examples = [], []
        for uid, example in example_dict.items():
            predictions.append(store.get(uid).decode('utf-8'))
            examples.append(example)
        torch.distributed.barrier()
        return predictions, [], examples
