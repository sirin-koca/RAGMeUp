from langchain.llms import HuggingFaceHub


def setup_llm():
    # Initialize the LLM from HuggingFace (NorMistral-warm-instruct for Norwegian)
    llm = HuggingFaceHub(repo_id="norallm/normistral-7b-warm-instruct",
                         model_kwargs={"temperature": 0.7, "max_length": 500})
    return llm


if __name__ == "__main__":
    llm = setup_llm()
    print("LLM is set up and ready.")
