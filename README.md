# HRTF-individualization
数据库：CIPIC数据库中有人体参数的37名听音人人体参数和hrir数据

模型：LightGBM模型

输入数据：人体参数数据，对应HRTF角度（水平角和高度角），HRTF频点的频率值

输出数据：HRTF频点的幅度值（也可以用于预测相位）

（如果使用深度学习的话，建议考虑同时输出幅度和相位）

由于现在算法比较往往是对整段HRTF使用谱失真(SD)比较，所以训练过程中将37名听音人中的30名作为训练集，剩下7名作为测试集。

引入频点作为输入的好处：偏于迁移到别的数据库，比较模型的泛化性能。

顺便推荐个性化HRTF讨论的QQ群：221606521
