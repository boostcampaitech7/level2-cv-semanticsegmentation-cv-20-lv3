import albumentations as A

class AlbumentationsTransform:
    def __init__(self, transform_config):
        self.aug_list = []
        for aug in transform_config:
            aug_class = getattr(A, aug["type"])
            self.aug_list.append(aug_class(**aug["params"]))
        self.transform = A.Compose(self.aug_list)

    def getTransform(self):
        return self.transform
        
class TransformSelector:
    def __init__(self, transform_type: str, transform_config: str=None):

        if transform_type in ["albumentations"]:
            self.transform_type = transform_type
            self.transform_config = transform_config
        
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self):
        
        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        if self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(transform_config=self.transform_config).getTransform()
        return transform