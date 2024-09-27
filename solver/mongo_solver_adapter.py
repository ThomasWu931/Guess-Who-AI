from .solver import *

class MongoSolver(Solver):
    def __init__(self, collection) -> None:
        super().__init__()
        self.collection = collection

    def get_images(self, image_ids) -> list[Image]:
        # raw_images = self.collection.find({"_id": {"$in": image_ids}})
        raw_images = list(self.collection.aggregate([{"$sample": {"size": 20}}]))
        images = []
        for image in raw_images:
            traits = {Trait(k): v["conf"] for k,v in image["traits"].items()}
            images.append(Image(str(image["_id"]), traits))
        return images

    def get_best_question_adapter(self, image_ids: list[str], depth=0) -> str:
        images = self.get_images(image_ids)
        question, _ = super().get_best_question(images, depth)
        return repr(question)