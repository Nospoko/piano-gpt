import json
from typing import Literal

import pandas as pd
import sqlalchemy as sa

from database.database_connection import database_cnx

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

validation_examples_dtype = {
    "example_id": sa.Integer,
    "generator_id": sa.Integer,
    "prompt_id": sa.Integer,
}

models_table = "models"
generators_table = "generators"
generations_table = "generations"
prompt_table = "prompt_notes"
validation_table = "validation_examples"

def insert_generated_notes(
    model: dict,
    prompt: dict,
    generator: dict,
    generated_notes: pd.DataFrame,
):
    generated_notes = generated_notes.to_json()
    prompt["prompt_notes"] = json.dumps(prompt["prompt_notes"])

    # Get or create IDs
    generator_id = register_generator(generator)
    prompt_id = register_prompt_notes(prompt)
    model_id = register_model(model_registration=model)

    # Check if the record already exists
    query = f"""
    SELECT generation_id
    FROM {generations_table}
    WHERE generator_id = {generator_id}
      AND prompt_id = {prompt_id}
      AND model_id = {model_id}
    """

    existing_record = database_cnx.read_sql(sql=query)

    if existing_record.empty:
        generation_data = {
            "generator_id": generator_id,
            "prompt_id": prompt_id,
            "model_id": model_id,
            "prompt_notes": prompt["prompt_notes"],
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

def get_generator(generator_id: int) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {generators_table}
    WHERE generator_id = {generator_id}
    """
    df = database_cnx.read_sql(sql=query)
    return df

def get_prompt(prompt_id: int) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {prompt_table}
    WHERE prompt_id = {prompt_id}
    """
    df = database_cnx.read_sql(sql=query)
    return df

def get_prompts_for_model(model_id: int) -> pd.DataFrame:
    query = f"""
    SELECT DISTINCT
        pn.*
    FROM {prompt_table} pn
    JOIN {generations_table} gn ON pn.prompt_id = gn.prompt_id
    WHERE gn.model_id = {model_id}
    """
    df = database_cnx.read_sql(sql=query)
    return df

def get_generator_for_model_and_prompt(model_id: int, prompt_id: int) -> pd.DataFrame:
    query = f"""
    SELECT DISTINCT g.*
    FROM {generators_table} g
    JOIN {generations_table} gn ON g.generator_id = gn.generator_id
    WHERE gn.model_id = {model_id} AND gn.prompt_id = {prompt_id}
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

def remove_validation_prompt(validation_prompt_id: int):
    query = f"""
    DELETE FROM {validation_table}
    WHERE example_id = {validation_prompt_id}
    """
    database_cnx.execute(query=query)

def get_model_predictions(
    model_filters: dict = None,
    prompt_filters: dict = None,
    generator_filters: dict = None,
) -> pd.DataFrame:
    base_query = f"""
    SELECT gn.*
    FROM {generations_table} gn
    JOIN {models_table} m ON gn.model_id = m.model_id
    JOIN {prompt_table} pn ON gn.prompt_id = pn.prompt_id
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

def get_all_generators() -> pd.DataFrame:
    query = f"SELECT * FROM {generators_table}"
    df = database_cnx.read_sql(sql=query)
    return df

def get_all_prompt_notes() -> pd.DataFrame:
    query = f"SELECT * FROM {prompt_table}"
    df = database_cnx.read_sql(sql=query)
    return df

def get_validation_examples_for_task(task: str) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {validation_table} ve
    JOIN {prompt_table} pn ON ve.prompt_id = pn.prompt_id
    JOIN {generators_table} g ON ve.generator_id = g.generator_id
    WHERE g.task = '{task}'
    """
    df = database_cnx.read_sql(sql=query)
    return df

def get_all_validation_prompts() -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {validation_table} ve
    JOIN {prompt_table} pn ON ve.prompt_id = pn.prompt_id
    JOIN {generators_table} g ON ve.generator_id = g.generator_id
    """
    df = database_cnx.read_sql(sql=query)
    return df

def register_model(model_registration: dict) -> int:
    query = f"""
    SELECT model_id
    FROM {models_table}
    WHERE name = '{model_registration['name']}'
    """

    existing_records = database_cnx.read_sql(sql=query)

    if not existing_records.empty:
        return existing_records.iloc[0]["model_id"]

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

def register_generator(generator: dict) -> int:
    query = f"""
    SELECT generator_id
    FROM {generators_table}
    WHERE generator_name = '{generator['generator_name']}'
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