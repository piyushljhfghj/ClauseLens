import pytest

from app.core.config import Settings


def make(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw",
    ["postgres://u:p@host:5432/db", "postgresql://u:p@host:5432/db"],
)
def test_database_url_is_converted_to_async_driver(raw: str) -> None:
    assert make(database_url=raw).database_url == "postgresql+asyncpg://u:p@host:5432/db"


def test_async_database_url_is_left_alone() -> None:
    url = "postgresql+asyncpg://u:p@host:5432/db"
    assert make(database_url=url).database_url == url


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = make(cors_origins=" http://localhost:3000/ , https://clauselens.vercel.app ,")
    assert settings.cors_origin_list == ["http://localhost:3000", "https://clauselens.vercel.app"]


def test_llm_configured_requires_all_three_values() -> None:
    assert not make().llm_configured
    assert not make(llm_base_url="https://x", llm_model="m", llm_api_key="  ").llm_configured
    assert make(llm_base_url="https://x", llm_model="m", llm_api_key="k").llm_configured


def test_api_key_is_never_rendered() -> None:
    settings = make(llm_api_key="sk-very-secret")
    assert "sk-very-secret" not in repr(settings)
    assert "sk-very-secret" not in str(settings.model_dump())
