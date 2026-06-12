from ragdoc.config import Settings
from ragdoc.generation.answerer import ExtractiveAnswerer
from ragdoc.pipeline import RagPipeline


def make_pipeline(embedder, **overrides):
    settings = Settings(**overrides) if overrides else Settings()
    return RagPipeline(settings=settings, embedder=embedder, answerer=ExtractiveAnswerer())


def test_end_to_end_ingest_and_ask(tmp_path, embedder):
    doc = tmp_path / "policy.md"
    doc.write_text(
        "# Company Policies\n\n"
        "Employees receive 25 days of paid vacation per year. "
        "Vacation requests must be submitted two weeks in advance.\n\n"
        "The office is closed on national holidays. "
        "Remote work is allowed up to three days per week."
    )
    pipeline = make_pipeline(embedder)
    count = pipeline.ingest([doc])
    assert count >= 1

    answer = pipeline.ask("How many vacation days do employees get?")
    assert "25" in answer.text
    assert answer.sources, "answer should carry source attribution"
    assert answer.sources[0].chunk.source.endswith("policy.md")


def test_index_persistence_roundtrip(tmp_path, embedder):
    doc = tmp_path / "facts.txt"
    doc.write_text("The warehouse is located in Pune. It operates from 9am to 6pm.")
    index_dir = tmp_path / "index"

    first = make_pipeline(embedder)
    first.ingest([doc])
    first.save_index(index_dir)

    second = make_pipeline(embedder)
    loaded = second.load_index(index_dir)
    assert loaded == len(first.retriever)
    answer = second.ask("Where is the warehouse located?")
    assert "Pune" in answer.text


def test_ask_with_empty_index_degrades_gracefully(embedder):
    pipeline = make_pipeline(embedder)
    answer = pipeline.ask("anything")
    assert "No relevant context" in answer.text
