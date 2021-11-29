![Arch](https://user-images.githubusercontent.com/77387431/143853833-f97cb2ed-f92d-4d8e-824d-4d28896f2bbf.png)


# Setup
This codebase is based on python3 (3.8.5). Other python versions should work seamlessly.

## Install Requirements
```shell
pip3 install -r requirements.txt
```

## Pip3 Freeze
In any circumstance, the requirements did not meet the setup then try the freeze based requirement
```shell
pip3 install -r pip3freeze.txt
```


## Get started

### Start
```shell
./run.sh articles/msft.txt
```

### To just generate the best summary
```shell
python3 best_summary.py articles/msft.txt
```

### Store reports
This is resource intensive and time consuming as each pegasus pretrained model is loaded and run.
So storing the created reports is advisable.

```shell
python3 best_summary.py articles/msft.txt > reports/msft.txt
```

### Look at reports
To see just the best summary only
```shell
tail -1 reports/msft.txt
```
