from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none

def preprocess_data(data, input_template=None, input_key=None, output_key=None) -> str:
    # custom dataset
    if input_key and output_key:
        prompt = data[input_key]
        response = data[output_key]
    else:
        # Dahoas/full-hh-rlhf
        if exist_and_not_none(data, "prompt") and exist_and_not_none(data, "chosen"):
            prompt = data["prompt"]
            response = data["chosen"]
            # tasksource/oasst1_pairwise_rlhf_reward
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ") + "\nAssistant: "
                )
            input_template = None  # do not modified with input template again
        # Open-Orca/OpenOrca
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"): 
            prompt = data["question"] # bak  # prompt = data["system_prompt"] + " " + data["question"]
            response = data["response"]
        else:
            raise ValueError("Unknown prompts dataset")

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, response


class BaseDataset(Dataset):
    """
    Dataset 
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        output_key = getattr(self.strategy.args, "output_key", None) 
        
        self.prompts = []
        self.responses = []
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, response = preprocess_data(data, input_template, input_key, output_key)

            self.prompts.append(prompt)
            self.responses.append(response)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.responses[idx]
