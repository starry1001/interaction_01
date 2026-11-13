import json
import datasets

class IWRBench(datasets.GeneratorBasedBuilder):
    def _info(self):
        features = datasets.Features({
            "id": datasets.Value("int64"),
            "classification": {
                "interaction_complexity": datasets.Value("string"),
                "visual_complexity": datasets.Value("string"),
                "domain": datasets.Value("string"),
                "sub_domain": datasets.Value("string"),
            },
            "recording_details": {
                "resolution": datasets.Value("string"),
                "environment": datasets.Value("string"),
            },
            "source_website": datasets.Value("string"),
            "video_path": datasets.Value("string"),
        })
        
        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        data_file = "data.jsonl"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_file},
            )
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                example = {
                    "id": data.get("id"),
                    "classification": data.get("classification"),
                    "recording_details": data.get("recording_details"),
                    "source_website": data.get("source_website"),
                    "video_path": data.get("video_path"),
                }
                
                yield data["id"], example

