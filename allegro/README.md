# Editing Nequip library so that it is compatible with torch geometric Data

Allegro takes in a custom class `AtomicDataDict` as input. 

To make the conversion from torch_geometric Data to `AtomicDataDict` work,
go to `/nequip/data/AtomicData.py` and


1. add `from torch_geometric.data import Data as GData` to the end
2. change `def to_AtomicDataDict` definition to

```   
    @staticmethod
    def to_AtomicDataDict(
        data: Union[Data, Mapping, GData], exclude_keys=tuple()
    ) -> AtomicDataDict.Type:
        if isinstance(data, Data) or isinstance(data, GData):
            keys = data.keys
        elif isinstance(data, Mapping):
            keys = data.keys()
        else:
            raise ValueError(f"Invalid data `{repr(data)}`")

        return {
            k: data[k]
            for k in keys
            if (
                k not in exclude_keys
                and data[k] is not None
                and isinstance(data[k], torch.Tensor)
            )
        }
```


