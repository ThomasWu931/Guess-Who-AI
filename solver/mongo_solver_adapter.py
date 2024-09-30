from .solver import *
from bson import ObjectId

class MongoSolver(Solver):
    def __init__(self, collection, image_ids: list[str], verbose: bool = False) -> None:
        self.collection = collection
        images = self._get_images(image_ids)
        super().__init__(images, verbose)

    def _get_images(self, image_ids) -> list[Image]:
        image_ids = [ObjectId(i) for i in image_ids]
        raw_images = list(self.collection.find({"_id": {"$in": image_ids}}))
        images = []
        for image in raw_images:
            traits = {Trait(k): v["conf"] for k,v in image["traits"].items()}
            images.append(Image(str(image["_id"]), traits))
        return images