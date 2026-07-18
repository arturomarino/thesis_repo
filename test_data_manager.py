from src.data_manager import DataManager

dm = DataManager("data/raw/copernicus.nc")

ds = dm.load()

print(ds)
print(ds.data_vars.keys())
