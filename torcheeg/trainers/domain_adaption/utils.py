from torch.utils.data import DataLoader


class DualDataLoader:

    def __init__(self, ref_dataloader: DataLoader,
                 other_dataloader: DataLoader):
        self.ref_dataloader = ref_dataloader
        self.other_dataloader = other_dataloader

    def __iter__(self):
        return self.dual_iterator()

    def __len__(self):
        return len(self.ref_dataloader)

    def dual_iterator(self):
        other_it = iter(self.other_dataloader)
        for data in self.ref_dataloader:
            try:
                data_ = next(other_it)
            except StopIteration:
                other_it = iter(self.other_dataloader)
                data_ = next(other_it)
            yield data, data_
