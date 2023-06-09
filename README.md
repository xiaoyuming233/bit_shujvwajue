两个代码文件分别对应两个数据集，其中main对应电影数据集，shujv2对应人体指标数据集
使用时将数据集与代码放在同一目录下，运行即可。有大量数据处理因为不同时使用而被放在了注释中
，执行时需要将对应内容取消注释

下面是数据集介绍

本次选择的两个数据集分别为 Movies Dataset from Pirated Sites 和 VitalDB

**Movies Dataset from Pirated Sites**

这个数据集是从一个盗版网站收集的，该网站的用户群每月约为 200 万。该数据包含来
自好莱坞、宝莱坞、动漫等所有行业的 20000 多部电影。数据包含如下字段：

id：电影的唯一 id

标题：电影名称

故事情节：对电影的简短描述

观看次数：每部电影的点击次数

下载：每部电影的下载次数

IMDb 评级：评级

适用于：R 级、PG-13 级等

语言：也可以是多种语言

行业：好莱坞、宝莱坞等。

发布日期：当电影发布在平台上时的日期

上映日期：电影在全球上映的时间

时长：以分钟为单位

导演：导演姓名

作者：所有作者的列表


**VitalDB**

在现代麻醉中，同时使用多种医疗设备来全面监测实时生命体征，以优化患者护理并改
善手术结果。VitalDB（生命体征数据库）是一个开放的数据集，专门用于促进与监测外科
患者生命体征相关的机器学习研究。该数据集包含 6388 例病例的高分辨率多参数数据，包
括 486451 个波形和数字数据轨迹，包括 196 个术中监测参数、73 个围手术期临床参数和 34
个时间序列实验室结果参数。

术中生命体征，如心电图、血压、经皮血氧饱和度和体温，是生理功能的客观测量，在
手术和麻醉期间由高灵敏度的患者监护仪跟踪。此前，我们开发了 Vital Recorder 程序，这
是一种数据采集软件，用于记录来自各种麻醉设备的时间同步高分辨率数据，包括患者监护
仪、麻醉机、脑监护仪、心脏监护仪、靶控输液泵和快速输注系统。同时应用于一名患者的
多个监测设备的所有参数都被记录为时间同步的数据轨道，并存储为单个病例文件。该程序
的自动记录功能使我们的三级大学医院能够大量收集术中生物信号。生命体征数据库
（VitalDB）使用（1）在日常手术和麻醉期间由生命记录仪程序自动记录的未识别病例文件，
以及（2）从我们的电子病历系统检索的围手术期患者信息构建。

与之前报道的公共多参数生物信号数据集不同，VitalDB 是第一个专门关注围手术期患
者护理的公共生物信号数据集中，其特征是包含多参数高分辨率波形和数字数据。自 2017
年首次发布 VitalDB 数据集以来，它已被用于各种大数据研究，如：基于动脉压波形的心输
出量算法的深度学习、基于深度学习的静脉麻醉药药代动力学药效学研究、双谱指数算法的
机器学习，术中双频谱 inde 关系的统计分析。
