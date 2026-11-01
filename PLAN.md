# 统一样本字典与流式数据处理重构

## 最终接口（2026-11-01）

用户明确不要求向后兼容。旧 `(X, y)`、变换关键字路由、apply_to_baseline、四个 before/after hook、Runner 绑定 IO、旧存储布局兼容入口最终全部删除。

- 样本为 `{'eeg': value, 'info': metadata, 'baseline': optional_value}`。
- Data 保留 set_records/read_record/fake_record/process_record。record 是输入文件或记录，一个 record 产生多个样本。
- `runner = MultiprocessRunner(num_worker=4)`；`runner(data)` 返回流式样本迭代器。
- `dataset = IODataset(eeg_io=..., info_io=...)`；`append(sample)` 和 `extend(samples)` 边消费边写入。
- `dataset[int]` 与遍历都返回独立样本字典；位置列表、切片、布尔数组返回子集，保留元数据类型与顺序。
- `dataset.info` 提供元数据 DataFrame；存储标识与 baseline 引用不进入用户样本。
- `with_transform(f).with_transform(g)` 等价于 `with_transform(Compose([f, g]))`，扁平追加、共享信号存储、不改原数据。
- 基础变换为 X -> X'；BaselineRemoval(baseline=...)(X)。样本变换负责选择字段。
- EEG IO 保留按 key/record 读取；Info IO 支持行/列读取；不另加组合 IO 或 IOData。
- 训练使用 `{'eeg': tensor, 'labels': tensor}`，现有 Trainer、示例同步迁移。

## 按依赖推进

- [x] 1. Dataset 核心：样本存取、筛选、复制隔离、Compose 与 with_transform 的统一契约；内存/持久 IO 回归。
- [x] 2. Transform：所有数组、张量、图、标签变换改为位置输入；baseline 在初始化绑定；删除目标路由与 apply_to_baseline；更新全部变换测试。
- [x] 3. Data：逐个迁移 34 个适配器，复用切分算法；baseline 内嵌样本；去除公开 key；fake 对照验证窗口、标签和数值。
- [x] 4. Runner：不持有 IO；串行逐条 yield；多进程有界队列按记录顺序输出，覆盖慢消费者、大记录、提前关闭、worker 异常、spawn/fork/forkserver；Joblib 明确限制缓冲，不依靠无限结果积压。
- [x] 5. 持久化流程：迁移已有 FileLock/manifest/临时目录发布；数据源指纹与 append/extend 更新语义分开；不把部分输出标记为完成；失败后可重试。
- [x] 6. Dataset 快捷入口、Pair/Concat、model_selection、Trainer 全面迁移；删除旧调用与兼容包装。
- [x] 7. 文档、教程、示例全部迁移；测试文件 mirror 源码，不新增零散回归测试文件。
- [x] 8. 验收：完整 CPU/fake suite、覆盖率门槛、格式、Sphinx、wheel 安装导入与完整处理/存储/训练 smoke。

每步完成后记录真实检查结果。未完成项不能被旧版 591 tests / 82.99% 结果覆盖。

## 关键测试

- append 后读回原数值与字符串 ID；修改返回值不能污染内存 IO 或 mmap。
- 子集索引：负索引、空集、重复位置、顺序、错误长度/类型的 mask；与 model_selection 共用视图。
- with_transform：顺序、嵌套 Compose 展平、原对象隔离、重读不累积修改、pickle 与 DataLoader worker。
- baseline：样本可选字段，不暴露独立辅助记录；重存储时引用正确，跨来源不会覆盖。
- 不收集整文件输出：消费者取得第一个样本时，该文件尚未全部处理；消费者关闭后 worker 退出。
- 数据集仅 fake/合成小文件；模型及训练仅 CPU。

## 使用

```python
data = DEAPData(root_path='./data_preprocessed_python', chunk_size=128)
dataset = IODataset(eeg_io=LMDBEEGIO('./cache/eeg'),
                    info_io=CSVInfoIO('./cache/info.csv'))
dataset.extend(MultiprocessRunner(num_worker=4)(data))
subject = dataset[dataset.info['subject_id'].eq('s01.dat').to_numpy()]
train = subject.with_transform(preprocess_sample).with_transform(prepare_training)
```

## 最终验证（2026-11-01）

- 完整 CPU/fake 测试：580 项全部通过，235.326 秒。
- 行与分支联合覆盖率：81.97%，现有门槛为 80%，没有降低门槛或添加覆盖率排除。
- 30 个原生 Data 适配器通过固定种子迁移前后首批样本对照：EEG 数值、窗口、标签与 baseline 无差异。其余通用/外部适配器使用合成输入或 stub 测试。
- 多进程测试覆盖 spawn/fork/forkserver、顺序、首样本流式输出、提前关闭、异常、LMDB 共享与重新打开；Joblib 线程测试通过。
- 220 个源码模块均有对应测试文件，测试结构没有缺失或额外文件。
- autopep8、isort 和 git diff --check 均通过。
- Sphinx HTML 构建成功，无 warning；依赖真实数据的教程未执行。
- wheel 在清理构建目录后重新构建、安装至独立目录；在源码目录外跑通多进程处理、LMDB/CSV 存储、筛选、with_transform 和 CPU 反向传播。
- 本地与服务器的 490 个 Python 文件校验值一致；wheel 中的 220 个源码文件校验值一致，无已删除模块残留。
- 验证日志保存在 `/Users/geovanni/Work/torcheeg-sample-refactor-validation-20261101/`。
- 实现分支：`refactor/data-runner-io`。

复现完整测试：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg python -m coverage run -m unittest discover -s test -t .
python -m coverage combine
python -m coverage report
```

## 实现取舍

- JoblibRunner 使用 Joblib 的线程后端和有界样本队列；进程并行使用 MultiprocessRunner。提前关闭会等待当前线程中的记录读取返回，Python 线程不能被强制终止。
- 显式 append/extend 是逐样本持久化，迭代失败前已写入的样本保留；Dataset 快捷入口的整次缓存构建通过临时目录和 manifest 发布，失败不发布完整缓存。
- MapChunk 的旧实现重复对整段 EEG 调用变换。本轮修复为处理实际窗口，并在原 test_concatenate.py 中验证每个窗口的数值；这是 API 迁移之外的一处行为修复。
- 真实数据集和 GPU 不用于本轮测试。文档构建关闭依赖真实数据的教程执行；另有独立数组示例和安装后完整处理链路的实际运行检查。
