import re
import json
import datetime

import pandas as pd
import sqlalchemy as sa

# TODO: from generation.generators import MidiGenerator
from database.database_connection import database_cnx


# TODO: make generation a package and remove the class below
class MidiGenerator:
    pass


prompt_dtype = {
    "prompt_id": sa.Integer,
    "midi_name": sa.String(255),
    "start_time": sa.Float,
    "end_time": sa.Float,
    "source": sa.JSON,
    "prompt_notes": sa.JSON,
}

model_dtype = {
    "model_id": sa.Integer,
    "base_model_id": sa.Integer,
    "name": sa.String(255),
    "milion_parameters": sa.Integer,
    "best_val_loss": sa.Float,
    "train_loss": sa.Float,
    "total_tokens": sa.Integer,
    "configs": sa.JSON,
    "training_task": sa.String(255),
    "wandb_link": sa.Text,
}

generator_dtype = {
    "generator_id": sa.Integer,
    "generator_name": sa.String(255),
    "generator_parameters": sa.JSON,
    "task": sa.String(255),
}

generations_dtype = {
    "generation_id": sa.Integer,
    "generator_id": sa.Integer,
    "prompt_id": sa.Integer,
    "model_id": sa.Integer,
    "prompt_notes": sa.JSON,
    "generated_notes": sa.JSON,
}

sources_dtype = {
    "source_id": sa.Integer,
    "source": sa.JSON,
    "notes": sa.JSON,
}

models_table = "models"
generators_table = "generators"
generations_table = "generations"
prompt_table = "prompt_notes"
sources_table = "sources"


def insert_generation(
    model_checkpoint: dict,
    model_name: str,
    generator: MidiGenerator,
    generated_notes: pd.DataFrame,
    prompt_notes: pd.DataFrame,
    source_notes: pd.DataFrame,
    source: dict,
):
    generated_notes = generated_notes.to_dict()
    prompt_notes = prompt_notes.to_dict()

    # Get or create IDs
    generator_id = register_generator_object(generator)
    _, model_id = register_model_from_checkpoint(
        checkpoint=model_checkpoint,
        model_name=model_name,
    )
    source_id = insert_source(
        notes=source_notes,
        source=source,
    )

    # Check if the record already exists
    query = f"""
    SELECT generation_id
    FROM {generations_table}
    WHERE generator_id = {generator_id}
      AND prompt_notes::text = '{json.dumps(prompt_notes)}'::text
      AND source_id = source_id
      AND model_id = {model_id}
    """
    existing_record = database_cnx.read_sql(sql=query)

    if existing_record.empty:
        generation_data = {
            "generator_id": generator_id,
            "model_id": model_id,
            "source_id": source_id,
            "prompt_notes": prompt_notes,
            "generated_notes": generated_notes,
        }
        # Insert the generation data
        df = pd.DataFrame([generation_data])
        database_cnx.to_sql(
            df=df,
            table=generations_table,
            dtype=generations_dtype,
            index=False,
            if_exists="append",
        )


def insert_source(source: dict, notes: pd.DataFrame) -> int:
    # Convert notes DataFrame to dict
    notes = notes.to_dict()

    # Check if the record already exists
    query = f"""
    SELECT source_id
    FROM {sources_table}
    WHERE source::text = '{json.dumps(source)}'::text
    """
    existing_record = database_cnx.read_sql(sql=query)

    if existing_record.empty:
        source_data = {
            "source": source,
            "notes": notes,
        }
        # Insert the source data
        df = pd.DataFrame([source_data])
        database_cnx.to_sql(
            df=df,
            table=sources_table,
            dtype=sources_dtype,
            index=False,
            if_exists="append",
        )

        # Fetch the inserted record's ID
        inserted_record = database_cnx.read_sql(sql=query)
        return inserted_record.iloc[0]["source_id"]
    else:
        return existing_record.iloc[0]["source_id"]


def get_generator(generator_id: int) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {generators_table}
    WHERE generator_id = {generator_id}
    """
    df = database_cnx.read_sql(sql=query)
    return df


def get_source(source_id: int) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {sources_table}
    WHERE source_id = {source_id}
    """
    df = database_cnx.read_sql(sql=query)
    return df


def get_models(model_name: str) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {models_table}
    WHERE name = '{model_name}'
    """
    df = database_cnx.read_sql(sql=query)
    return df


def get_model_id(model_name: str) -> int:
    query = f"""
    SELECT model_id
    FROM {models_table}
    WHERE name = '{model_name}'
    """
    df = database_cnx.read_sql(sql=query)
    if len(df) == 0:
        return None
    else:
        return df.iloc[-1]["model_id"]


def purge_model(model_name: str):
    notes_query = f"""
    DELETE FROM {generations_table}
    WHERE model_id IN (
        SELECT model_id FROM {models_table} WHERE name = '{model_name}'
    )
    """
    database_cnx.execute(notes_query)

    model_query = f"""
    DELETE FROM {models_table}
    WHERE name = '{model_name}'
    """
    database_cnx.execute(model_query)


def get_model_predictions(
    model_filters: dict = None,
    prompt_filters: dict = None,
    generator_filters: dict = None,
) -> pd.DataFrame:
    base_query = f"""
    SELECT gn.*
    FROM {generations_table} gn
    JOIN {models_table} m ON gn.model_id = m.model_id
    JOIN {generators_table} g ON gn.generator_id = g.generator_id
    WHERE 1=1
    """

    if model_filters:
        for key, value in model_filters.items():
            base_query += f" AND m.{key} = '{value}'"

    if prompt_filters:
        for key, value in prompt_filters.items():
            base_query += f" AND pn.{key} = '{value}'"

    if generator_filters:
        for key, value in generator_filters.items():
            base_query += f" AND g.{key} = '{value}'"

    df = database_cnx.read_sql(sql=base_query)
    return df


def get_unique_values(column, table):
    query = f"SELECT DISTINCT {column} FROM {table} ORDER BY {column}"
    df = database_cnx.read_sql(sql=query)
    return df[column].dropna().tolist()


def get_all_models() -> pd.DataFrame:
    query = f"SELECT * FROM {models_table}"
    df = database_cnx.read_sql(sql=query)
    return df


def select_models_with_generations() -> pd.DataFrame:
    query = """
    SELECT
        m.model_id,
        m.base_model_id,
        m.name,
        m.milion_parameters,
        m.best_val_loss,
        m.train_loss,
        m.iter_num,
        m.total_tokens,
        m.training_task,
        m.wandb_link,
        m.created_at
    FROM models m
    WHERE m.model_id IN (
        SELECT DISTINCT model_id
        FROM generations
    )
    ORDER BY m.model_id
    """
    df = database_cnx.read_sql(sql=query)

    # Fetch configs separately
    configs_query = """
    SELECT model_id, configs
    FROM models
    WHERE model_id IN (
        SELECT DISTINCT model_id
        FROM generations
    )
    """
    configs_df = database_cnx.read_sql(sql=configs_query)

    # Merge the results
    df = pd.merge(df, configs_df, on="model_id", how="left")

    return df


def get_all_generators() -> pd.DataFrame:
    query = f"SELECT * FROM {generators_table}"
    df = database_cnx.read_sql(sql=query)
    return df


def get_all_prompt_notes() -> pd.DataFrame:
    query = f"SELECT * FROM {prompt_table}"
    df = database_cnx.read_sql(sql=query)
    return df


def register_model_from_checkpoint(
    checkpoint: dict,
    model_name: str,
):
    # Hard-coded for the specific naming style
    milion_parameters = model_name.split("-")[2][:-1]
    init_from = checkpoint["config"]["init_from"]
    base_model_id = None
    if init_from != "scratch":
        base_model_id = get_model_id(model_name=init_from)

    model_registration = {
        "name": model_name,
        "milion_parameters": milion_parameters,
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "iter_num": checkpoint["iter_num"],
        "training_task": checkpoint["config"]["task"],
        "configs": checkpoint["config"],
    }
    if "wandb" in checkpoint.keys():
        model_registration |= {"wandb_link": checkpoint["wandb"]}
    if "total_tokens" in checkpoint.keys():
        model_registration |= {"total_tokens": checkpoint["total_tokens"]}
    if "train_loss" in checkpoint.keys():
        model_registration |= {"train_loss": float(checkpoint["train_loss"])}
    if base_model_id is not None:
        model_registration |= {"base_model_id": base_model_id}

    model_id = register_model(model_registration=model_registration)

    return model_registration, model_id


def register_model(model_registration: dict) -> int:
    query = f"""
    SELECT model_id
    FROM {models_table}
    WHERE name = '{model_registration['name']}'
    AND total_tokens = '{model_registration['total_tokens']}'
    """

    existing_records = database_cnx.read_sql(sql=query)

    if not existing_records.empty:
        return existing_records.iloc[0]["model_id"]

    # Extract datetime from model name
    date_match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})", model_registration["name"])
    if date_match:
        created_at = datetime.strptime(date_match.group(1), "%Y-%m-%d-%H-%M")
    else:
        created_at = datetime.now()  # Use current time if pattern not found
    model_registration |= {"created_at": created_at}

    df = pd.DataFrame([model_registration])
    database_cnx.to_sql(
        df=df,
        table=models_table,
        dtype=model_dtype,
        index=False,
        if_exists="append",
    )

    df = database_cnx.read_sql(sql=query)
    return df.iloc[0]["model_id"]


def register_generator_object(generator: MidiGenerator) -> int:
    generator_desc = {
        "generator_name": generator.__class__.__name__,
        "task": generator.task,
        "generator_parameters": generator.parameters,
    }
    return register_generator(generator=generator_desc)


def register_generator(generator: dict) -> int:
    parameters = json.dumps(generator["generator_parameters"])
    generator_name = generator["generator_name"]
    task = generator["task"]

    query = f"""
    SELECT generator_id
    FROM {generators_table}
    WHERE generator_name = '{generator_name}'
    AND generator_parameters::text = '{parameters}'::text
    AND task = '{task}'
    """
    existing_records = database_cnx.read_sql(sql=query)
    if not existing_records.empty:
        return existing_records.iloc[0]["generator_id"]

    df = pd.DataFrame([generator])
    database_cnx.to_sql(
        df=df,
        table=generators_table,
        dtype=generator_dtype,
        index=False,
        if_exists="append",
    )
    df = database_cnx.read_sql(sql=query)
    return df.iloc[0]["generator_id"]


def register_prompt_notes(prompt_notes: dict) -> int:
    query = f"""
    SELECT prompt_id
    FROM {prompt_table}
    WHERE start_time = {prompt_notes['start_time']}
      AND end_time = {prompt_notes['end_time']}
      AND midi_name = '{prompt_notes['midi_name']}'
    """
    existing_records = database_cnx.read_sql(sql=query)

    if not existing_records.empty:
        return existing_records.iloc[0]["prompt_id"]

    df = pd.DataFrame([prompt_notes])
    database_cnx.to_sql(
        df=df,
        table=prompt_table,
        dtype=prompt_dtype,
        index=False,
        if_exists="append",
    )
    df = database_cnx.read_sql(sql=query)
    return df.iloc[0]["prompt_id"]


def get_model_tasks(model_id: int) -> list:
    query = f"""
    SELECT DISTINCT g.task
    FROM {generations_table} gn
    JOIN {generators_table} g ON gn.generator_id = g.generator_id
    WHERE gn.model_id = {model_id}
    ORDER BY g.task
    """
    df = database_cnx.read_sql(sql=query)
    return df["task"].tolist()


def remove_models_without_generations():
    query = f"""
    SELECT DISTINCT model_id
    FROM {generations_table}
    """
    models_with_generations = database_cnx.read_sql(sql=query)

    model_ids_with_generations = models_with_generations["model_id"].tolist()

    delete_query = f"""
    DELETE FROM {models_table}
    WHERE model_id NOT IN ({','.join(map(str, model_ids_with_generations))})
    """
    database_cnx.execute(delete_query)
