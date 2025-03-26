import re


class MusicManager:
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

    def __init__(
        self,
        max_n_notes: int,
        max_absolute_time: float = 120.0,
    ):
        self.max_n_notes = max_n_notes
        self.max_absolute_time = max_absolute_time
        self.composer_regex_map = self.create_composer_regex_map()

    @property
    def absolute_time_tokens(self) -> list[str]:
        tokens = []
        # From seconds to 100ms steps
        max_time_steps = int(10 * self.max_absolute_time)
        for it in range(max_time_steps + 1):
            time_step = it / 10
            token = self.get_absolute_time_token(time_step)
            tokens.append(token)

        return tokens

    @property
    def n_note_tokens(self) -> list[str]:
        # NOTE: 0 is a valid number of notes
        tokens = [self.get_n_notes_token(n_notes) for n_notes in range(self.max_n_notes + 1)]
        return tokens

    @property
    def tokens(self) -> list[str]:
        return self.dataset_tokens + self.composer_tokens + self.n_note_tokens + self.absolute_time_tokens

    def create_composer_regex_map(self) -> dict[re.Pattern, str]:
        regex_map: dict[re.Pattern, str] = {}
        for full_name, token in self.composer_token_map.items():
            names = full_name.split()
            surname = names[-1]
            pattern = re.compile(rf"\b{re.escape(surname)}\b", re.IGNORECASE)
            regex_map[pattern] = token
        return regex_map

    def get_dataset_token(self, piece_source: dict) -> str:
        dataset_name = piece_source.get("dataset")

        for dataset_token in self.dataset_tokens:
            dataset_token_name = dataset_token[1:-1]
            if dataset_token_name.lower() == dataset_name.lower():
                return dataset_token

        # FIXME Our internal dataset is the only one without the name
        # stored as part of the source. This should change with the next
        # dataset version, then we can add <UNKNOWN_DATASET> here
        return "<PIANO_FOR_AI>"

    def get_composer_token(self, composer: str) -> str:
        # TODO This should be more refined - we know that composer
        # informaion is stored in many ways across different datasets
        # and we should use that knowledge:
        # def get_composer_token(dataset_name: str, piece_source: dict): ...
        matches: list[tuple[re.Match, str]] = [
            (match, token) for pattern, token in self.composer_regex_map.items() if (match := pattern.search(composer))
        ]

        if len(matches) == 1:
            return matches[0][1]

        return "<UNKNOWN_COMPOSER>"

    def get_n_notes_token(self, n_notes: int) -> str:
        return f"<N_NOTES_{n_notes}>"

    def get_absolute_time_token(self, music_time: float) -> list[str]:
        # Clip time to the maximum value
        # TODO spend some time trying to understand the duration
        # distribution of records in the Piano dataset

        # oh god why https://stackoverflow.com/a/944712
        is_nan = music_time != music_time
        if is_nan:
            # NaN can happen if we get an empty target (possible in Chords task)
            music_time = 0

        music_time = min(music_time, self.max_absolute_time)

        # Arbitrary resolution choice: 100ms
        music_time = int(10 * round(music_time, 1))
        token = f"<MUSIC_TIME_{music_time}>"
        return token
