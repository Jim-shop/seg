# BlastSeg

基于PyTorch的胚胎分割神经网络。


## 数据集

1. 请从[此处](http://vault.sfu.ca/index.php/s/066vGJfviJMYuP6)下载胚胎分割数据集。  
2. 将`BlastsOnline`文件夹解包出来，整个放在`data`文件夹下。  
3. 然后在`data`文件夹下执行`DataProcess.py`文件，即可在`ProcessData`文件夹下创建所需的标签图片。  
4. 再在`data`文件夹下执行`DataGrouping.py`文件，即可随机切分训练集测试集并在`data`文件夹下得到三个`.txt`清单文件。

> **此时`data`文件夹结构应当是这样：**
> 
> - 📂`data`
>   - 📁`BlastOnline`
>   - 📁`ProcessData`
>   - 🐍`DataGrouping.py`
>   - 🐍`DataProcess.py`
>   - 📄`test_list.txt`
>   - 📄`train_list.txt`
>   - 📄`val_list.txt`


## 依赖

```pip
cv2
```

## 使用方法

### 训练

训练处理程序是`train.py`。  
训练参数在`.yaml`文件中配置。
  
项目根目录命令行输入：

```shell
    python3 train.py ["配置文件.yaml"]
```

即可开始训练。

> **注意：**
> 
> `["配置文件.yaml"]`请更换成实际`.yaml`文件名，方框表示可选。
  
