from src.modules.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()

for page in pipeline.run("tests/assets/1.pdf"):
    print(
        page["page_index"],
        page["status"],
        page["width"],
        page["height"],
        page["metadata"]
    )
