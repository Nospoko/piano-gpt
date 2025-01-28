import re

dataset_tokens = [
    "<MAESTRO>",
    "<PIJAMA>",
    "<VGMIDI>",
    "<MUSIC-NET>",
    "<PIANO-MIDI-DE>",
    "<LAKH-LMD-FULL>",
    "<GIANT-MIDI>",
    "<IMSLP>",
    "<ATEPP-1.1>",
    "<PIANO_FOR_AI>",
]
composer_tokens = [
    "<SCRIABIN>",
    "<FRANCK>",
    "<MOZART>",
    "<CHOPIN>",
    "<MENDELSSON>",
    "<LISZT>",
    "<SCHUBERT>",
    "<BRAHMS>",
    "<HAYDN>",
    "<BEETHOVEN>",
    "<BALAKIREV>",
    "<SCHUMANN>",
    "<RACHMANIOFF>",
    "<UNKNOWN_COMPOSER>",
    "<BACH>",
]

piano_task_tokens = [
    "<CLEAN_TIME>",
    "<CLEAN_EVERYTHING>",
    "<CLEAN_VOLUME>",
    "<CLEAN_PITCH>",
    "<LOW_FROM_MEDIAN>",
    "<HIGH_FROM_MEDIAN>",
    "<ABOVE_LOW_QUARTILE>",
    "<BELOW_LOW_QUARTILE>",
    "<ABOVE_HIGH_QUARTILE>",
    "<BELOW_HIGH_QUARTILE>",
    "<MIDDLE_QUARTILES>",
    "<EXTREME_QUARTILES>",
    "<LOUD>",
    "<SOFT>",
    "<ABOVE_VERY_SOFT>",
    "<VERY_SOFT>",
    "<VERY_LOUD>",
    "<BELOW_VERY_LOUD>",
    "<MODERATE_VOLUME>",
    "<EXTREME_VOLUME>",
    "<CLEAN>",
    "<NOISY_VOLUME>",
    "<NOISY_PITCH>",
    "<NOISY_START_TIME>",
    "<NOISY_TIME>",
    "<NOISY>",
]

composer_token_map: dict[str, str] = {
    "Alexander Scriabin": "<SCRIABIN>",
    "César Franck": "<FRANCK>",
    "Wolfgang Amadeus Mozart": "<MOZART>",
    "Frédéric Chopin": "<CHOPIN>",
    "Felix Mendelssohn": "<MENDELSSON>",
    "Franz Liszt": "<LISZT>",
    "Franz Schubert": "<SCHUBERT>",
    "Johannes Brahms": "<BRAHMS>",
    "Joseph Haydn": "<HAYDN>",
    "Ludwig van Beethoven": "<BEETHOVEN>",
    "Mily Balakirev": "<BALAKIREV>",
    "Robert Schumann": "<SCHUMANN>",
    "Sergei Rachmaninoff": "<RACHMANIOFF>",
    "Johann Sebastian Bach": "<BACH>",
}


def get_dataset_token(piece_source: dict) -> str:
    dataset_name = piece_source.get("dataset")

    for dataset_token in dataset_tokens:
        dataset_token_name = dataset_token[1:-1]
        if dataset_token_name.lower() == dataset_name.lower():
            return dataset_token

    # FIXME Our internal dataset is the only one without the name
    # stored as part of the source. This should change with the next
    # dataset version, then we can add <UNKNOWN_DATASET> here
    return "<PIANO_FOR_AI>"


def create_composer_regex_map() -> dict[re.Pattern, str]:
    regex_map: dict[re.Pattern, str] = {}
    for full_name, token in composer_token_map.items():
        names = full_name.split()
        surname = names[-1]
        pattern = re.compile(rf"\b{re.escape(surname)}\b", re.IGNORECASE)
        regex_map[pattern] = token
    return regex_map


composer_regex_map: dict[re.Pattern, str] = create_composer_regex_map()


def get_composer_token(composer: str) -> str:
    # TODO This should be more refined - we know that composer
    # informaion is stored in many ways across different datasets
    # and we should use that knowledge:
    # def get_composer_token(dataset_name: str, piece_source: dict): ...
    matches: list[tuple[re.Match, str]] = [
        (match, token) for pattern, token in composer_regex_map.items() if (match := pattern.search(composer))
    ]

    if len(matches) == 1:
        return matches[0][1]
    return "<UNKNOWN_COMPOSER>"
