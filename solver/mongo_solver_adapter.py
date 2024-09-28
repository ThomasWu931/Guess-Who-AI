from .solver import *

class MongoSolver(Solver):
    def __init__(self, collection, image_ids: list[str], questions: list[Question], answers: list[Answer]) -> None:
        self.collection = collection
        images = self._get_images(image_ids)
        super().__init__(images, questions, answers)

    def _get_images(self) -> list[Image]:
        # raw_images = self.collection.find({"_id": {"$in": image_ids}}) # TODO: UNCOMMENT LATER
        raw_images = list(self.collection.aggregate([{"$sample": {"size": 20}}]))
        images = []
        for image in raw_images:
            traits = {Trait(k): v["conf"] for k,v in image["traits"].items()}
            images.append(Image(str(image["_id"]), traits))
        return images