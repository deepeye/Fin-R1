# 金融领域蒸馏数据集（Financial-R1-Distill-Data）

## 目录 <a name="toc"></a>

1. [简介](#intro)
2. [数据集详情与数据量分布](#datasets)
3. [任务设置](#tasks)
4. [使用说明](#usage)

---

## 简介 <a name="intro"></a>

欢迎来到Financial-R1-Distill-Data！

Financial-R1-Distill-Data是面向专业金融场景开发的高质量指令微调数据集，是用Deepseek-R1（完整版）提炼出来的SFT数据集。该数据集总规模约30k条，中英混合，覆盖金融垂直领域的多维度专业知识。数据源来自Ant_Finance、FinanceIQ、FinanceQT、ConvFinQA、TFNS、Finance-Instruct-500k、FinPEE、FinCorpus、FinCUGE九大数据集。按照内容整理为以下四大模块：

- 金融代码
- 金融专业知识
- 金融非推理类业务知识
- 金融推理类业务知识

![Financial-R1-Distill-Data总览](.frame_cn.png)

[返回顶部](#toc)

---



## 数据集详情与分布 <a name="datasets"></a>

### 金融代码
Financial-R1-Distill-Data金融专业代码集是专为金融领域垂直打造的高精度代码生成基准测试集，聚焦量化交易、投资策略设计与金融工程开发三大核心场景。该数据集精选60条经过严格验证的金融代码生成任务，数据源主体来自FinanceQT集，经人工筛选、领域适配性重构及标准化处理后形成结构化语料，它包含60条高质量的金融代码生成问题。以下是金融专业代码集的数据示例：

>
> **问题:**  
>
> Read the context below and answer the questions.  
> Context: You are a helpful assistant for quant trading.  
> Question: Please provide me the code for a basic template algorithm that includes futures with daily resolution and extended market hours.  
> 
> **模型答案:**  
>
> To create a custom data type history algorithm using QuantConnect, follow this detailed implementation. The example below demonstrates how to ingest alternative data from a CSV file, fetch its historical values, and use it in a trading strategy.  
>  
> ```python
> # Import required modules
> from datetime import datetime, timedelta
> from QuantConnect import *
> from QuantConnect.Algorithm import *
> from QuantConnect.Data import *
> from QuantConnect.Data.Custom import *
> from QuantConnect.Python import *
> 
> # 1. Define Custom Data Type
> class AlternativeDataSet(PythonData):
>     """Custom dataset representing alternative data (e.g., sentiment scores)"""
>     
>     def GetSource(self, config: SubscriptionDataConfig, date: datetime, isLiveMode: bool) -> SubscriptionDataSource:
>         # Path to CSV file in QuantConnect's Data Directory under 'alternative' folder
>         return SubscriptionDataSource(
>             "alternative/sentiment_scores.csv",
>             SubscriptionTransportMedium.LocalFile,
>             FileFormat.Csv
>         )
>     
>     def Reader(self, config: SubscriptionDataConfig, line: str, date: datetime, isLiveMode: bool) -> AlternativeDataSet:
>         # Parse CSV line: Format -> Date, Symbol, Score
>         if not line.strip() or line.startswith("Date"):  # Skip headers/empty lines
>             return None
>             
>         parsed = line.split(',')
>         data_point = AlternativeDataSet()
>         data_point.Symbol = config.Symbol
>         data_point.Time = datetime.strptime(parsed[0], "%Y-%m-%d")
>         data_point["Score"] = float(parsed[2])  # Store custom field
>         data_point.Value = data_point["Score"]  # Default value property
>         return data_point
> 
> # 2. Implement Trading Algorithm
> class CustomDataHistoryAlgorithm(QCAlgorithm):
>     def Initialize(self):
>         self.SetStartDate(2023, 1, 1)
>         self.SetCash(100000)
>         self.current_holdings = 0
>         
>         # Add custom data subscription
>         self.sentiment_symbol = self.AddData(
>             AlternativeDataSet,
>             "SENTIMENT",  # Unique identifier
>             Resolution.Daily
>         ).Symbol
>         
>         # Fetch 90-day historical data
>         history = self.History[AlternativeDataSet](
>             self.sentiment_symbol,
>             timedelta(days=90),
>             Resolution.Daily
>         )
>         
>         # Validate and process history
>         if len(history) == 0:
>             raise ValueError("No historical data loaded")
>             
>         # Calculate average sentiment score
>         scores = [x.Value for x in history]
>         self.avg_score = sum(scores) / len(scores)
>         self.Debug(f"Historical Average Sentiment: {self.avg_score}")
>         
>         # Initialize rolling window for live updates
>         self.score_window = RollingWindow[float](30)
>         
>         # Schedule daily analysis
>         self.Schedule.On(
>             self.DateRules.EveryDay(self.sentiment_symbol),
>             self.TimeRules.AfterMarketOpen("SPY", 0),
>             self.AnalyzeSentiment
>         )
> 
>     def OnData(self, data: Slice):
>         # Update rolling window with real-time data
>         if self.sentiment_symbol in data:
>             self.score_window.Add(data[self.sentiment_symbol].Value)
> 
>     def AnalyzeSentiment(self):
>         # Ensure sufficient data points
>         if not self.score_window.IsReady:
>             return
>             
>         # Calculate metrics
>         current_score = self.score_window[0]
>         moving_avg = sum(self.score_window) / self.score_window.Size
>         
>         # Trading logic
>         if current_score > moving_avg and current_score > self.avg_score:
>             if self.current_holdings < 1:
>                 self.SetHoldings("SPY", 1.0)
>                 self.current_holdings = 1
>         elif self.current_holdings > 0:
>             self.Liquidate("SPY")
>             self.current_holdings = 0
> 
> # 3. Key Features Demonstrated
> # - Custom data ingestion from CSV
> # - Historical data fetching for initialization
> # - Rolling window for real-time analysis
> # - Scheduled event for daily rebalancing
> # - Integration with equity trading (SPY)
> ```
> 
> **Implementation Notes:**  
> 1. **Data Parsing**: The `AlternativeDataSet` class handles CSV parsing with proper error checking.  
> 2. **Historical Context**: Uses 90-day history to calculate baseline sentiment averages.  
> 3. **Live Updates**: Implements rolling window for real-time score tracking.  
> 4. **Robust Checks**: Validates data availability and handles edge cases.  
> 5. **Scheduled Execution**: Separates analysis from market events for consistent daily evaluation.  
> 
> This implementation shows a complete pipeline from custom data ingestion to strategy execution while meeting QuantConnect's framework requirements. The code is structured for easy adaptation to different alternative data sources by modifying the `AlternativeDataSet` class and trading logic.


### 金融专业知识
Financial-R1-Distill-Data金融专业知识数据集是面向金融垂直领域深度优化的知识评估与训练资源库，其内容涵盖选择题、计算题、文本问答题三大核心题型，覆盖保险精算、股票交易策略、微观经济分析、宏观经济建模等细分场景，总规模约5800条高质量数据。该数据集通过融合多源异构金融数据资源，构建核心数据源与特性如下：
>
Finance-Instruct（5,650条）：是一个专为金融领域高级语言模型训练设计的综合性数据集，包含超过 50 万条条目，覆盖金融问答、推理、情感分析、主题分类、多语言命名实体识别（NER）和对话 AI 等多个任务。其特点包括广泛的数据覆盖、多轮对话支持、多样化数据来源、RAG 格式数据、去重与预处理、XBRL 标记等。本项目中聚焦其中的金融逻辑推理与开放问答任务，涵盖企业财报分析、投资决策推演等高阶能力评估问题；

FinPEE（174条）：由项目组联合金融科研团队从近年国内顶级高校金融考研真题中蒸馏提取，覆盖衍生品定价、资本资产定价模型（CAPM）、财务报表重构等核心计算题型，每题均经持证分析师进行考点映射与难度分级，强化理论与实务的结合。

以下是金融专业知识Finance-Instruct的问题示例：

>
> **问题:**  
>  
>Determine the economic impact of various environmental policies and regulations on different industries and sectors of the economy. Explain how a carbon tax, cap and trade system, pollution standards or incentives for green technology would affect business costs, consumer prices and economic growth in the short and long term. Provide specific examples and details of companies, industries and economic trends that would be impacted.
>
> **模型答案:** 
>
>Okay, so I need to figure out the economic impact of different environmental policies on various industries and sectors. Let me start by breaking down each policy mentioned: carbon tax, cap and trade, pollution standards, and green tech incentives.
>
>First, a carbon tax. That's a direct tax on the carbon emissions companies produce, right? So if a company emits a lot of CO2, they have to pay more. This would increase their operating costs. For industries like fossil fuels—coal, oil, gas—this would be a big hit. Their production costs go up, which might lead them to either absorb the cost (lowering profits) or pass it on to consumers through higher prices. For example, an oil company like ExxonMobil would face higher costs, so gasoline prices might rise. Consumers paying more could reduce demand, maybe leading to decreased sales for these companies. However, in the long term, companies might invest in cleaner technologies to reduce their tax burden. But in the short term, there's a cost increase and possible inflation in energy prices.
>
>Then there's cap and trade. This is where the government sets a cap on emissions and companies can trade permits. So companies that reduce emissions can sell their permits to others. This creates a market for carbon credits. Industries with high emissions, like manufacturing or utilities, would need to buy more permits if they can't reduce emissions. For example, a power company like Duke Energy might have to purchase additional permits, increasing their costs. However, companies that can reduce emissions cheaply might profit by selling permits. This system could lead to innovation as companies look for cost-effective ways to cut emissions. In the short term, similar to carbon tax, costs might rise, but over time, the market adjusts, possibly leading to more efficient emission reductions compared to a fixed tax.
>
>Pollution standards are regulations that set limits on pollutants. For instance, the EPA might mandate that all coal plants reduce sulfur dioxide emissions. Companies would have to invest in scrubbers or other technologies. This increases capital expenditures. The automotive industry faced this with emission standards, leading to catalytic converters. So, car manufacturers like Toyota or Ford have to invest in R&D to meet these standards, which increases production costs. These costs might be passed to consumers, making cars more expensive. In the short term, this could reduce sales, but long term, it pushes the industry towards cleaner technology, possibly opening new markets for electric vehicles or hybrids.


以下是金融知识数据集FinPEE的问题示例:

>
>**问题:**
>
>请根据所给信息解答以下金融计算题 某公司的股票期望收益率是16%,标准差0.20,无风险利率是6%。A女士打技货400元于股票,600元于无风险资产。计算其资产组合的期望收益率。
>
>**回答:**
>
> <think>
> 好的，我现在需要解决这个金融计算题。题目是说，某公司的股票期望收益率是16%，标准差0.20，无风险利率是6%。A女士投资了400元于股票，600元于无风险资产。要计算她的资产组合的期望收益率。
>
> 首先，我应该回忆一下资产组合期望收益率的计算方法。一般来说，组合的期望收益率是各个资产期望收益率的加权平均，权重就是各个资产在总资产中的比例。对吧？
>
> 那A女士的总投资是多少呢？她投了400元在股票，600元在无风险资产，所以总资产应该是400 + 600 = 1000元。没错，总共有1000元。
>
> 接下来，我要确定每个资产在组合中的权重。股票的权重就是股票投资额除以总资产，也就是400/1000，对吗？同样，无风险资产的权重就是600/1000。这里我需要确认一下，是否每个资产的权重确实是按投资比例计算的，是的，应该是这样。
>
> 那股票的权重是400/1000=0.4，也就是40%。无风险资产的权重是600/1000=0.6，也就是60%。对吧？
>
> 然后，股票的期望收益率是16%，无风险资产的收益率是6%。所以组合的期望收益率应该是各个资产的权重乘以其期望收益率，然后相加。也就是0.4*16% + 0.6*6%。对吧？
>
> 现在具体计算的话，0.4乘以16%等于多少呢？0.4*16=6.4，所以这部分是6.4%。然后0.6乘以6%是3.6%。然后加起来的话，6.4%+3.6%=10%。所以组合的期望收益率应该是10%。
>
> 不过，我需要再检查一下计算是否正确。比如权重是否正确，总资产是不是1000元，没错。股票部分0.4乘以16%，得到6.4%。无风险部分是0.6乘以6%，得到3.6%。合计10%。没错，看起来是对的。
>
> 可能有人会问，这里标准差是不是用不上？因为题目只问期望收益率，标准差可能跟风险有关，但这里不需要计算风险，所以不需要用到标准差的数据。所以应该没问题。
>
> 所以，最终的答案应该是10%。也就是组合的期望收益率是10%。
> </think>
> <answer>
>
> 首先，计算A女士的总投资金额：>[ 400元（股票） + 600元（无风险资产） = 1000元 ]
>
> 接着，确定各资产的权重：>- 股票的权重为：>[ frac{400}{1000} = 0.4 ]
>
> - 无风险资产的权重为：>[ frac{600}{1000} = 0.6 ]
>
> 然后，计算资产组合的期望收益率。组合的期望收益率是各资产期望收益率的加权平均：
>
> [ E(R_p) = (0.4 times 16%) + (0.6 times 6%) ]
>
> [ = 6.4% + 3.6% = 10% ] 
>
> 最终，资产组合的期望收益率为：
>
> boxed{10%}
>
> </answer>


### 金融非推理类业务知识
Financial-R1-Distill-Data金融非推理类业务知识数据集是面向金融领域基础认知能力评估与训练的专项数据集，聚焦选择题与判断题两大题型，覆盖金融业务中知识记忆、合规判断、语义理解等非推理核心能力。该数据集通过整合多源权威数据，构建了覆盖金融全场景的标准化评测体系，总规模约16K条高质量数据，其核心架构与特性如下：

FinanceIQ（1038条）：这是一个专注于金融领域的中文知识评估数据集，旨在全面评估大语言模型在金融场景下的通用能力、专业知识和场景应用能力。它包含10个金融大类和36个小类，精选其10个金融大类中的监管合规、金融工具定义、市场基础规则等非推理单选题。该数据集的推出旨在填补金融领域大模型评估体系的空白，帮助优化模型在金融业务中的表现。

Ant-Finance（619条）：源自蚂蚁集团业务脱敏数据集，聚焦金融实操场景，强化信息安全性判断（如客户数据脱敏规则）、用户评价情感辨析、业务流程合规性审查、金融名词槽位识别（如ABS与MBS术语辨析）等实战化题型。内容涉及保险、会计、投融资、金融产品等多个方面。

FinCorpus（15,000条）：FinCorpus数据集是一个高质量的中文金融语料库，由度小满团队开源，旨在支持金融领域的自然语言处理任务。该数据集包含上市公司公告、金融资讯、新闻文章以及金融试题等多种类型的数据，总规模大约60GB。经清洗后保留上市公司公告解析、金融新闻事件链推理等结构化知识单元；

以下为金融非推理类业务知识 FinanceIQ 数据示例：

>
> **问题:**  
>
> 关于税务机关实施税收保全措施的时间限制，通常不能超过半年。当面临重大案件需要延长时，应该先征得批准。那么具备批准权的税务机关是哪一级别的呢？  选项:  A: 省级税务局    B: 县级税务局    C: 国家税务总局    D: 市级税务局  
> 
> **回答:**  
>
> <think>
>嗯，这个问题是关于税务机关实施税收保全措施的时间限制以及延长批准权限的。我需要仔细分析题目中的关键点。题目说，通常税收保全措施不能超过半年，重大案件需要延长时需要先征得批准。问题是问哪个级别的税务机关有批准权。
>
>首先，根据《税收征收管理法》及其实施细则的规定，税收保全措施的一般期限是不得超过六个月。如果遇到重大案件需要延长，必须向上级税务机关申请批准。那么这里的上级机关具体是哪一级呢？
>
>我记得相关的法规里可能提到过，比如在《税收征管法实施细则》中，延长税收保全措施需要经过国家税务总局或者省级税务局的批准？或者是不是省级税务局？可能选项中的正确答案是A选项，省级税务局。
>
>不过，我需要再仔细回忆一下。比如，根据税收征管法的规定，税务机关采取税收保全措施的期限一般不得超过六个月；重大案件需要延长的，应当报国家税务总局批准。但这里可能有变化，或者不同的法规条款？
>
>或者是不是根据最新的实施细则，延长税收保全措施需要报经省级税务局批准？比如，之前的学习中有没有类似的知识点？
>
>例如，国家税务总局可能负责更宏观的管理，而具体的延长审批可能由省级税务局处理。比如，在一般的行政层级中，省级税务局是地方上的最高税务机关，可能对于本省内的重大案件有审批权。而如果涉及全国性的重大案件，可能需要国家税务总局批准，但题目中的选项里有A和C选项，而问题可能更倾向于常规的重大案件，因此可能需要省级税务局的批准。
>
>因此，可能正确答案是选项A：省级税务局。而选项C是国家税务总局，可能在某些特别的情况下才需要，但题目中的选项是常规情况下的延长，因此选A。
></think>
><answer>
>
>根据《税收征收管理法实施细则》第八十八条的规定，税务机关采取税收保全措施的期限一般不得超过6个月；对重大案件需要延长的，应当报经国家税务总局批准。因此，在重大案件需要延长税收保全措施时，批准权属于国家税务总局。
>
>boxed{C：国家税务总局}
>
></answer>

以下为金融非推理类业务知识 Ant-Finance 数据示例：

>
> **问题:**  
> 什么是保险营销   选项:  A: 保险营销是指保险公司通过暴力手段强制向消费者销售保险产品的活动。   B: 保险营销是指保险公司利用虚假广告和欺骗手段来推销低质量的保险产品的行为。   C: 保险经营机构及相关人员进行保险理念宣传、保险产品销售及相关服务的活动。 D: 保险营销是指保险公司通过贿赂、商业贿赂等非法手段来推销保险产品的活动。
> 
> **回答:**  
> 好，我现在遇到了一个关于保险营销的选择题，需要仔细分析每个选项的内容，然后找出正确的答案。题目问的是“什么是保险营销”，给出的选项是A、B、C、D四个。我需要逐一分析每个选项，看看哪个最符合保险营销的定义。
>
>    首先，我应该回忆一下保险营销的基本概念。保险营销，顾名思义，应该是指保险公司或其相关机构为了推广保险产品而进行的各种活动。这可能包括宣传保险理念、销售保险产品、提供相关的售后服务等。正确的定义应该正面描述这些活动，而不是涉及非法或不道德的手段。
>
>    接下来，逐个看选项：
>
>    选项A：保险营销是指保险公司通过暴力手段强制向消费者销售保险产品的活动。这里提到了“暴力手段”和“强制销售”，这显然是不合法的行为，而正常的营销应该是基于自愿原则的，所以A选项应该不正确。
>
>    选项B：保险营销是指保险公司利用虚假广告和欺骗手段来推销低质量的保险产品的行为。这里的关键词是“虚假广告”、“欺骗手段”和“低质量产品”，这些都是负面甚至违法的行为，而营销本身并不一定是非法的，因此B选项也不正确。
>
>    选项C：保险经营机构及相关人员进行保险理念宣传、保险产品销售及相关服务的活动。这里提到了宣传理念、销售产品和相关服务，这些都是保险营销的正常组成部分，符合我对保险营销的理解，所以C选项可能是正确的。
>
>    选项D：保险营销是指保险公司通过贿赂、商业贿赂等非法手段来推销保险产品的活动。贿赂属于非法手段，正常的营销活动不应该包括这些，所以D选项不正确。
>
>    再进一步验证一下。根据保险行业的常识，营销活动是合法且需要遵循行业规范的，包括教育消费者、推广产品、提供咨询和服务等。而选项C中的描述符合这一点，而其他选项都涉及违法行为或负面手段，显然不符合实际定义。
>
>    所以，正确的答案应该是选项C。

以下为金融非推理类业务知识 FinCorpus 数据示例：

> **问题:**  
> 某酒厂为增值税一般纳税人，2019年7月销售白酒4000斤，取得销售收入13560元（含增值税）。已知白酒消费税定额税率为0.5元/斤，比例税率为20%。该酒厂7月应缴纳的消费税税额为（ ）元。
A、2000
B、2400
C、4400
D、5000
>
> **回答:**  
>
><think>
>嗯，我现在得仔细看看这个问题。题目是关于酒厂缴纳消费税的计算，对吧？我记得消费税有从价和从量两种方式，白酒可能同时适用这两种。那题目里的数据是销售了4000斤白酒，取得销售收入13560元，含增值税。消费税的定额税率是0.5元每斤，比例税率是20%。选项是四个数，我得算出正确的答案。
>
>首先，增值税一般纳税人，所以他们的销售价格是含增值税的。但消费税的计税依据应该是不含增值税的销售额，对吧？所以首先要把含税收入转换成不含税的。增值税税率是多少呢？一般纳税人通常适用13%的税率，这是2019年时候的情况吗？可能吧。假设这里的增值税税率是13%。
>
>所以，不含税销售额应该是13560元除以（1+13%），这样算出不含税销售额。然后从价税部分用这个销售额乘以20%，加上从量税的4000斤乘以0.5元每斤。这两部分加起来就是应缴纳的消费税总额。
>
>让我一步一步来。首先计算不含税销售额。13560除以1.13。那13560 ÷ 1.13等于多少呢？先计算一下。13乘1000是13000，13560-13000是560，所以大概是12000左右吗？或者更准确的话，用计算器算的话，13560 ÷ 1.13等于12000元。对吗？因为1.13乘12000等于13560，对的，没错。所以不含税销售额是12000元。
>
>然后从价税部分是12000乘以20%，也就是2400元。从量税部分是4000斤乘以0.5元，等于2000元。两者相加2400+2000=4400元。所以答案应该是选项C，4400元。那选项中C选项是4400，所以应该选C。
>
>不过等等，我要再检查一下步骤有没有哪里出错。有没有可能题目中的增值税计算有别的处理方式？比如是否应该直接用销售收入计算从价税，而不用扣除增值税？这个时候可能需要再确认消费税的计税依据是否含增值税。根据消费税的规定，消费税是价内税，而增值税是价外税，所以计算消费税的销售额应该不含增值税。所以这里的处理是对的，先要价税分离，然后计算从价部分。所以这样计算是正确的。
>
>所以答案应该是选项C，4400元。
></think>
><answer>>
>酒厂作为增值税一般纳税人，销售白酒取得的收入13560元为含增值税销售额。计算消费税时需将含税收入转换为不含税销售额，再分别计算从价税和从量税，具体步>骤如下：
>
>1. **计算不含税销售额**  
   >增值税税率为13%，不含税销售额为：  
   >[
   frac{13560}{1 + 13%} = frac{13560}{1.13} = 12000  text{元}
   ]
>
>2. **计算从价税部分**  
   >比例税率为20%，因此从价税为：  
   >[
   12000 times 20% = 2400  text{元}
   ]
>
>3. **计算从量税部分**  
   >定额税率为0.5元/斤，销售量为4000斤，因此从量税为：  
   >[
   4000 times 0.5 = 2000  text{元}
   ]
>
>4. **合计消费税总额**  
   >将两部分税额相加：  
   >[
   2400 + 2000 = 4400  text{元}
   ]
>
>综上，酒厂7月应缴纳的消费税税额为 **4400元**，对应选项C。
>
>boxed{C、4400}
></answer>
>

### 金融推理类业务知识
Financial-R1-Distill-Data金融推理类业务知识集是面向金融领域复杂逻辑推理与数值计算能力的高质量数据集，涵盖表格推理、对话问答、事件因果分析等多模态任务类型，总规模达15,000条数据。该数据集通过整合FinQA、ConvFinQA、TFNS、FinCUGE四大核心模块，构建了覆盖金融全场景的推理能力评测与训练体系，其核心架构与特性如下：

FinQA（2,900条）：聚焦金融表格数据驱动的数值推理任务，模拟真实世界中分析师对财务报表重组、财务比率交叉验证、现金流预测等场景的挑战专注于金融领域数值推理的数据集，旨在提升AI在处理金融数据时的数值理解能力。包含详细的表格、文本描述以及相关问题和答案。
单步计算：如“根据利润表计算毛利率”
多步推导：如“基于现金流量表与资产负债表推算自由现金流（FCF）”
逻辑判断：如“若应收账款周转率下降，是否意味着公司回款能力减弱”
    
ConvFinQA（3800条）:采用轮次级对话问答数据，模拟投顾服务、投资者会议等多轮交互场景，强化模型对上下文连贯性与数值推理一致性的处理能力。题型设计如下：
数值推理：如“根据上一轮对话中的EPS增长率，推算下一财年净利润”
逻辑推演：如“若美联储加息，对港股科技股估值有何影响”
知识融合：如“结合公司公告与行业数据，分析本次并购的协同效应”

TFNS（1200条）：
聚焦金融事件因果关系推理，涵盖政策影响分析、市场波动归因、企业行为推演等复杂任务。对关键事件附加时间轴标注，支持时序推理任务。题型设计如下：
事件链构建：如“分析央行降准对银行间市场利率的传导路径”
影响评估：如“评估贸易摩擦对出口型企业股价的短期与长期影响”
反事实推理：如“若未发生某次并购，公司当前市值会如何变化”

FinCUGE（2,000条）：整合金融领域NLP专项任务，涵盖事件问答、因果推断、主体抽取等推理密集型场景。题型设计如下：
事件问答：如“某公司被立案调查的原因是什么”
因果推断：如“分析某次股价暴跌与公司高管减持的关联性”

以下为金融推理类业务知识 FinQA 数据示例：

>
> **问题:**  
>
>Please answer the question based on all the information provided below:
>
>Document: MRO/2008/page_119.pdf
>
>ID: MRO/2008/page_119.pdf-3
>
>Previous Text:
marathon oil corporation notes to consolidated financial statements 7 .
dispositions outside-operated norwegian properties 2013 on october 31 , 2008 , we closed the sale of our norwegian outside-operated properties and undeveloped offshore acreage in the heimdal area of the norwegian north sea for net proceeds of $ 301 million , with a pretax gain of $ 254 million as of december 31 , 2008 .
pilot travel centers 2013 on october 8 , 2008 , we completed the sale of our 50 percent ownership interest in ptc .
sale proceeds were $ 625 million , with a pretax gain on the sale of $ 126 million .
immediately preceding the sale , we received a $ 75 million partial redemption of our ownership interest from ptc that was accounted for as a return of investment .
operated irish properties 2013 on december 17 , 2008 , we agreed to sell our operated properties located in ireland for proceeds of $ 180 million , before post-closing adjustments and cash on hand at closing .
closing is subject to completion of the necessary administrative processes .
as of december 31 , 2008 , operating assets and liabilities were classified as held for sale , as disclosed by major class in the following table : ( in millions ) 2008 .
>
>Table Data:
>[['( in millions )', '2008'], ['current assets', '$ 164'], ['noncurrent assets', '103'], ['total assets', '267'], ['current >liabilities', '62'], ['noncurrent liabilities', '199'], ['total liabilities', '261'], ['net assets held for sale', '$ 6']]
>
>Following Text:
>8 .
>discontinued operations on june 2 , 2006 , we sold our russian oil exploration and production businesses in the khanty-mansiysk region of western siberia .
under the terms of the agreement , we received $ 787 million for these businesses , plus preliminary working capital and other closing adjustments of $ 56 million , for a total transaction value of $ 843 million .
proceeds net of transaction costs and cash held by the russian businesses at the transaction date totaled $ 832 million .
a gain on the sale of $ 243 million ( $ 342 million before income taxes ) was reported in discontinued operations for 2006 .
income taxes on this gain were reduced by the utilization of a capital loss carryforward .
exploration and production segment goodwill of $ 21 million was allocated to the russian assets and reduced the reported gain .
adjustments to the sales price were completed in 2007 and an additional gain on the sale of $ 8 million ( $ 13 million before income taxes ) was recognized .
the activities of the russian businesses have been reported as discontinued operations in the consolidated statements of income and the consolidated statements of cash flows for 2006 .
revenues applicable to discontinued operations were $ 173 million and pretax income from discontinued operations was $ 45 million for 2006. .
>
>Question: what is working capital for 2008?"
>
> **回答:**
>
><think>
>Okay, let's see. The question is asking for the working capital for 2008. Hmm, working capital is typically calculated as current assets minus current liabilities. So I need to find those two numbers from the provided data.
>
>Looking at the table data given in the document. The table has entries for current assets and current liabilities. Let me check that again. The table lists:
>
>[['( in millions )', '2008'], ['current assets', '$ 164'], ['noncurrent assets', '103'], ['total assets', '267'], ['current liabilities', '62'],
>['noncurrent liabilities', '199'], ['total liabilities', '261'], ['net assets held for sale', '$ 6']]
>
>So for 2008, current assets are $164 million, and current liabilities are $62 million. Therefore, working capital should be 164 minus 62. Let me compute that: 164 - 62 equals 102. So the working capital would be $102 million.
>
>Wait a second, but the question is specifically about 2008. The table is under the section titled "operated irish properties 2013" but the date mentioned in the text is December 17, 2008, and the table's data is as of December 31, 2008. So the table is relevant to 2008. The note says "as of December 31, 2008, operating assets and liabilities were classified as held for sale, as disclosed by major class in the following table". So the current assets and current liabilities here are part of the held for sale classification. Does that affect the calculation of working capital for the entire company, or is this just a segment?
>
>Hmm, working capital for the company as a whole would include all current assets and all current liabilities. However, this table might only be showing the assets and liabilities held for sale related to the Irish properties. Wait, but the previous text mentions that they agreed to sell the operated Irish properties, and as of December 31, 2008, the operating assets and liabilities were classified as held for sale. The table is showing those held-for-sale assets and liabilities. So the current assets here ($164 million) and current liabilities ($62 million) are part of the held-for-sale classification. But does that mean that the entire company's working capital is just this, or is this a subset?
>
>Wait, the question is "what is working capital for 2008?" without specifying a segment. So if the entire company's working capital is required, we might need more information. But the problem is that the document provided only includes this table under the "operated Irish properties" section. The rest of the text talks about other dispositions, like Norwegian properties, Pilot Travel Centers, and then the Irish properties. The table's data is specifically about the net assets held for sale related to the Irish properties.
>
>But if the question is about the working capital for the entire Marathon Oil Corporation for 2008, we might not have all the necessary data. However, the user has only provided this document snippet. The previous text and the following text mention other sales and discontinued operations, but the table here is part of the "operated Irish properties" note. Wait, the note says "as of December 31, 2008, operating assets and liabilities were classified as held for sale, as disclosed by major class in the following table". So this table is showing the current and noncurrent assets and liabilities held for sale. The "net assets held for sale" is $6 million, which is total assets ($267) minus total liabilities ($261) equals $6 million.
>
>But if the user is asking for the working capital (current assets minus current liabilities) for 2008, and the only data provided here is the held-for-sale assets and liabilities, then perhaps the answer is based on this table. However, working capital typically refers to the company's overall current assets minus current liabilities. But if the question is about the working capital related to the assets held for sale, then it's 164 - 62 = 102. But I need to make sure.
>
>Wait, the question is general: "what is working capital for 2008?" Given that the user provided only this document, which includes this table, maybe the intended answer is to use the current assets and current liabilities from this table. But the table is specifically for the assets held for sale, not the entire company. However, the problem might be that the user expects the answer from the given data, regardless of context. Alternatively, maybe the question is a trick, because the table's current assets and liabilities are part of the held-for-sale, and thus excluded from working capital. Wait, no. When assets are held for sale, they are still part of the company's current assets if they are expected to be sold within a year. But the classification as held for sale would mean they are current assets. Similarly, the liabilities directly associated would be current liabilities. However, the question is about working capital for 2008. If the entire company's working capital isn't provided, but only the held-for-sale part, maybe the answer is $102 million based on the table. But I need to check.
>
>Wait, let me re-read the note. The text says, "as of December 31, 2008, operating assets and liabilities were classified as held for sale, as disclosed by major class in the following table". So the table is showing the major classes of assets and liabilities held for sale. The question is about working capital for 2008. If working capital is current assets minus current liabilities for the entire company, but the table only shows the held-for-sale portion, then the answer might not be here. However, since this is the only data provided, maybe the answer is intended to be $102 million. But I need to be cautious. Alternatively, maybe the question is referring to the working capital of the Irish properties held for sale, which would be the 164 - 62 = 102. But the question isn't specifying that. Hmm. This is confusing.
>
>Alternatively, perhaps the "net assets held for sale" is $6 million, but that's total assets minus total liabilities (267 - 261 = 6). But working capital is just current assets minus current liabilities. So regardless of the held-for-sale classification, if the question is asking for the working capital related to those assets, then it's 164 - 62 = 102. But if it's asking for the entire company's working capital, then we don't have enough data. But given that the user provided only this information, they probably expect us to use the table's current assets and current liabilities. So the answer is $102 million. Therefore, the working capital for 2008 is $102 million.
></think>
><answer>
>The working capital for 2008 is calculated as **current assets minus current liabilities** using the data provided in the table under the "operated Irish properties" section. 
>
>From the table:
>- **Current assets** = $164 million  
>- **Current liabilities** = $62 million  
>
>**Calculation**:  
>[
text{Working Capital} = text{Current Assets} - text{Current Liabilities} = 164 - 62 = 102
]
>
>**Verification**:  
>The table explicitly lists assets and liabilities classified as held for sale as of December 31, 2008. While the broader context of Marathon Oil Corporation's total working capital might require additional data, the question is answered using the **specific figures provided in the document** for assets and liabilities held for sale. The result aligns with standard accounting definitions and the given inputs.
>
>boxed{102}
></answer>
>

以下为金融推理类业务知识 Convfinqa 数据示例：

>
> **问题:** 
>
>Read the following texts and table with financial data from an S&P 500 earnings report carefully.
>Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided.
>The net decrease in the 2016 effective tax rate was due , in part , to the 2016 asset impairments in the u.s . and to the current year benefit related to a restructuring of one of our brazilian businesses that increases tax basis in long-term assets . further , the 2015 rate was impacted by the items described below . see note 20 2014asset impairment expense for additional information regarding the 2016 u.s . asset impairments . income tax expense increased $ 101 million , or 27% ( 27 % ) , to $ 472 million in 2015 . the company's effective tax rates were 41% ( 41 % ) and 26% ( 26 % ) for the years ended december 31 , 2015 and 2014 , respectively . the net increase in the 2015 effective tax rate was due , in part , to the nondeductible 2015 impairment of goodwill at our u.s . utility , dp&l and chilean withholding taxes offset by the release of valuation allowance at certain of our businesses in brazil , vietnam and the u.s . further , the 2014 rate was impacted by the sale of approximately 45% ( 45 % ) of the company 2019s interest in masin aes pte ltd. , which owns the company 2019s business interests in the philippines and the 2014 sale of the company 2019s interests in four u.k . wind operating projects . neither of these transactions gave rise to income tax expense . see note 15 2014equity for additional information regarding the sale of approximately 45% ( 45 % ) of the company 2019s interest in masin-aes pte ltd . see note 23 2014dispositions for additional information regarding the sale of the company 2019s interests in four u.k . wind operating projects . our effective tax rate reflects the tax effect of significant operations outside the u.s. , which are generally taxed at rates lower than the u.s . statutory rate of 35% ( 35 % ) . a future proportionate change in the composition of income before income taxes from foreign and domestic tax jurisdictions could impact our periodic effective tax rate . the company also benefits from reduced tax rates in certain countries as a result of satisfying specific commitments regarding employment and capital investment . see note 21 2014income taxes for additional information regarding these reduced rates . foreign currency transaction gains ( losses ) foreign currency transaction gains ( losses ) in millions were as follows: . <table class='wikitable'><tr><td>1</td><td>years ended december 31,</td><td>2016</td><td>2015</td><td>2014</td></tr><tr><td>2</td><td>aes corporation</td><td>$ -50 ( 50 )</td><td>$ -31 ( 31 )</td><td>$ -34 ( 34 )</td></tr><tr><td>3</td><td>chile</td><td>-9 ( 9 )</td><td>-18 ( 18 )</td><td>-30 ( 30 )</td></tr><tr><td>4</td><td>colombia</td><td>-8 ( 8 )</td><td>29</td><td>17</td></tr><tr><td>5</td><td>mexico</td><td>-8 ( 8 )</td><td>-6 ( 6 )</td><td>-14 ( 14 )</td></tr><tr><td>6</td><td>philippines</td><td>12</td><td>8</td><td>11</td></tr><tr><td>7</td><td>united kingdom</td><td>13</td><td>11</td><td>12</td></tr><tr><td>8</td><td>argentina</td><td>37</td><td>124</td><td>66</td></tr><tr><td>9</td><td>other</td><td>-2 ( 2 )</td><td>-10 ( 10 )</td><td>-17 ( 17 )</td></tr><tr><td>10</td><td>total ( 1 )</td><td>$ -15 ( 15 )</td><td>$ 107</td><td>$ 11</td></tr></table> total ( 1 ) $ ( 15 ) $ 107 $ 11 _____________________________ ( 1 ) includes gains of $ 17 million , $ 247 million and $ 172 million on foreign currency derivative contracts for the years ended december 31 , 2016 , 2015 and 2014 , respectively . the company recognized a net foreign currency transaction loss of $ 15 million for the year ended december 31 , 2016 primarily due to losses of $ 50 million at the aes corporation mainly due to remeasurement losses on intercompany notes , and losses on swaps and options . this loss was partially offset by gains of $ 37 million in argentina , mainly due to the favorable impact of foreign currency derivatives related to government receivables . the company recognized a net foreign currency transaction gain of $ 107 million for the year ended december 31 , 2015 primarily due to gains of : 2022 $ 124 million in argentina , due to the favorable impact from foreign currency derivatives related to government receivables , partially offset by losses from the devaluation of the argentine peso associated with u.s . dollar denominated debt , and losses at termoandes ( a u.s . dollar functional currency subsidiary ) primarily associated with cash and accounts receivable balances in local currency , 2022 $ 29 million in colombia , mainly due to the depreciation of the colombian peso , positively impacting chivor ( a u.s . dollar functional currency subsidiary ) due to liabilities denominated in colombian pesos , 2022 $ 11 million in the united kingdom , mainly due to the depreciation of the pound sterling , resulting in gains at ballylumford holdings ( a u.s . dollar functional currency subsidiary ) associated with intercompany notes payable denominated in pound sterling , and .
Question: what was the effective tax rate in 2015?
>
> **回答:** 
>
><think>
>Okay, let's tackle this question. The user is asking for the effective tax rate in 2015. I need to go through the provided text and data to find that information.
>
>First, I'll start by scanning the text for mentions of \"effective tax rate\" and the year 2015. The text mentions that the effective tax rates were 41% for 2015 and 26% for 2014. Specifically, the sentence says: \"the company's effective tax rates were 41% (41%) and 26% (26%) for the years ended December 31, 2015 and 2014, respectively.\" So that seems straightforward. The answer is 41%, but I need to make sure there's no other context that might complicate this.
>
>Wait, the user also provided a table with financial data. Let me check if the table has any tax rate information. The table is about foreign currency transaction gains and losses. The columns are for 2016, 2015, and 2014, but the data there is about losses and gains in different countries. There's a line for AES Corporation and other countries, but nothing directly about tax rates. The tax rate info must be in the text, not the table.
>
>The text also discusses reasons for changes in the effective tax rates between the years, like asset impairments, restructuring benefits, and other tax-related items. For 2015, it mentions that the increase in the effective tax rate was due to nondeductible impairments and withholding taxes, offset by some allowances. However, the question is just asking for the rate itself, not the reasons behind it. So even though there's a lot of detail about why the rate changed, the actual number is given clearly as 41% for 2015.
>
>I should double-check if there's any other mention of 2015 tax rates. Let me re-read the relevant paragraph. The user wrote: \"income tax expense increased $101 million, or 27% (27%), to $472 million in 2015. The company's effective tax rates were 41% (41%) and 26% (26%) for the years ended December 31, 2015 and 2014, respectively.\" Yes, that's the key part. The effective tax rate in 2015 is explicitly stated as 41%. The rest of the information explains factors affecting the rate but doesn't change the actual percentage provided.
>
>Therefore, the answer should be 41%. The table doesn't include tax rates, so the text is the sole source here. No calculations are needed since the rate is directly given. Just need to confirm there's no typo or conflicting data elsewhere. But scanning through, all mentions of 2015's tax rate align with 41%. So the answer is definitely 41%.
></think>
><answer>
>
>To determine the effective tax rate in 2015, we analyze the provided text. The passage explicitly states:  
*\"the company's effective tax rates were 41% (41%) and 26% (26%) for the years ended December 31, 2015 and 2014, respectively.\"*  
>
>This sentence directly specifies the **2015 effective tax rate as 41%** without requiring further calculation. The surrounding context discusses factors influencing the rate (e.g., nondeductible impairments, foreign taxes, and valuation allowances), but these details do not alter the explicitly stated rate. The table provided focuses on foreign currency transaction gains/losses and does not contain tax rate data, reinforcing that the textual statement is the primary source.  
>
>Thus, the effective tax rate in 2015 is definitively **41%**.  
>
>boxed{41%}
></answer>

以下为金融推理类业务知识 TFNS 数据示例：
> **问题:** 
>Below is a finance-related tweet. Please analyze its sentiment.
>    \"0\" represents \"Bearish\"
>    \"1\" represents \"Bullish\"
>    \"2\" represents \"Neutral\"
>
>  Text: Lindsay Corp beats by $0.04",
>
> **答案:** 
> <think>
>Okay, let me start analyzing the sentiment of this tweet about Lindsay Corp. The user wants me to determine if it's Bearish, Bullish, or Neutral. The tweet says, \"$LNN: Lindsay Corp beats by $0.04, reports revs in-line https://t.co/DRuphR9XZM\". 
>>
>First, I need to break down the components. The company beat earnings estimates by $0.04 per share. That's a positive sign because beating earnings typically indicates good performance, which could make investors optimistic. The revenue part is reported as in-line, meaning it met expectations but didn't exceed them. 
>
>Now, sentiment in finance tweets often hinges on earnings and revenue results. Beating earnings is generally bullish because it shows the company is performing better than analysts expected. However, revenue being in-line might temper that enthusiasm a bit. The absence of a revenue beat could mean that while profits are up, maybe due to cost-cutting or one-time gains, the core business isn't growing as much.
>
>But the key here is that the earnings beat is a strong positive signal. The market often reacts more to earnings surprises than revenue, especially if the revenue was already anticipated. The fact that they mention \"beats\" first might lead the reader to focus on the positive aspect. However, the revenue not beating could lead to a mixed reaction. 
>
>Also, the URL link might contain more details, but since it's a shortened link, I can't access it. So I have to rely solely on the text provided. 
>
>Considering the options: 0 (Bearish), 1 (Bullish), 2 (Neutral). If both earnings and revenue were missed, it would be bearish. If both beat, clearly bullish. Here, one beat and one in-line. Neutral might be a consideration, but earnings beats are usually more impactful. However, some might argue that an earnings beat without a revenue beat could indicate cost management rather than top-line growth, which may not be as bullish. 
>
>But the overall tone here is more positive than negative. The headline starts with a beat, which is good news. The market often reacts positively to such news. Therefore, I'm leaning towards Bullish (1). However, I need to check if there's any nuance I'm missing. Maybe the revenue in-line is a negative in context? But in-line means they met expectations, so it's not bad. 
>
>Alternatively, maybe the market expected a revenue beat as well, so only meeting revenue could be seen as a slight negative. But given that earnings are a key metric, the beat there might overshadow the in-line revenue. 
>
>Another angle: the company's stock might already have priced in the earnings expectations. But since they beat it, it's still positive. 
>
>So after considering all factors, I think the sentiment is Bullish (1).
></think>
><answer>
>\\boxed{1}
></answer>"
>



以下为金融推理类业务知识 FinCUGE 数据示例：
>
> **问题:**
> 
> 巨人网络股份回购事件对应的占公司总股本比例是什么？原标题：巨人网络(20.750,-0.15,-0.72%)(002558.SZ)：截至8月底累计耗资4.26亿元回购1.2183%股份  
>
> **回答:** 
><think>
>好的，我现在要解决用户的问题，也就是找出巨人网络股份回购事件中，回购的股份占公司总股本的比例是多少。用户给的信息是一个新闻段落，我需要仔细分析里面的内容。
>
>首先，我看看用户提供的原文。里面提到：“截至2020年8月31日，公司通过集中竞价方式累计回购股份数量为2466.3916万股，占公司总股本的1.2183%。”这里的“占公司总股本的1.2183%”看起来就是问题所要的答案。不过，我需要确认这个百分比是否确实对应的是总股本的比例，而不是其他什么比例。
>
>接下来，我需要检查是否有其他数据可能混淆这个比例。例如，新闻中还提到了最高成交价、最低成交价和总金额，但这些可能和比例无关。用户的问题明确询问的是占总股本的比例，所以应该直接看是否有明确的百分比数字。根据给出的数据，1.2183%已经被明确标注为占总股本的比例，所以这应该就是正确的答案。
>
>不过，为了确保万无一失，我再仔细看一下原文的结构。该新闻的标题和正文中都提到了1.2183%的比例，并且明确说明是“占公司总股本的”。因此，可以确定这个数值就是用户需要的信息。没有其他可能的数值干扰，比如总金额或者股数，但问题问的是比例，所以答案应该是这个百分比。
>
>最后，确认一下格式要求，需要用\\boxed{}包裹答案，所以正确的输出应该是\\boxed{1.2183%}。整个过程应该没有问题，数据来源明确，比例直接给出，无需计算，直接提取即可。
>
>
></think>
>
><answer>
>
>根据任务指令和提供的输入，巨人网络在股份回购事件中累计回购的股份占公司总股本的比例已在原文中明确提及：“占公司总股本的1.2183%”
>最终答案：\\boxed{1.2183\\%}
></answer>"
>

[返回顶部](#toc)


---

## 使用说明 <a name="usage"></a>

### 1. 项目结构

```
Financial_Dataset/
├── data/             # 数据集原始文件
├── scripts/          # 脚本文件
├── logs/             # 日志文件
├── output/           # 输出文件
├── README.md         # 项目说明
└── requirements.txt  # 依赖环境
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行示例

```bash
bash scripts/run_training.sh
```

### 4. 输出说明

- 模型训练结果存放于`output/`
- 运行日志位于`logs/`

[返回顶部](#toc)
