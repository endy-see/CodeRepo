# AKShare 数据接口测试报告

> 生成时间: 2026-03-20 18:13
> AKShare 版本: 1.18.11
> 测试总数: 1068 | ✅ 成功: 594 | ⏱ 超时: 352 | ❌ 失败: 101 | ⊘ 跳过: 20

## 目录

- [股票数据](#股票数据) (211个接口)
- [宏观数据](#宏观数据) (108个接口)
- [指数数据](#指数数据) (62个接口)
- [基金数据](#基金数据) (48个接口)
- [期权数据](#期权数据) (34个接口)
- [期货数据](#期货数据) (30个接口)
- [债券数据](#债券数据) (24个接口)
- [其他](#其他) (24个接口)
- [另类数据](#另类数据) (23个接口)
- [现货数据](#现货数据) (7个接口)
- [货币数据](#货币数据) (4个接口)
- [外汇数据](#外汇数据) (4个接口)
- [新闻资讯](#新闻资讯) (4个接口)
- [申万指数](#申万指数) (4个接口)
- [波动率/学术](#波动率/学术) (3个接口)
- [加密货币](#加密货币) (3个接口)
- [工具箱](#工具箱) (1个接口)
- [分类统计汇总](#分类统计汇总)
- [失败接口列表](#失败接口列表)

---

## 股票数据

✅ 211 成功 | ❌ 33 失败 | ⏱ 158 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `stock_a_all_pb` | 5142x8 | 无需参数 | date, middlePB, equalWeightAveragePB, close, quantileInAllHistoryMiddlePB, quantileInRecent10YearsMiddlePB, quantileInAllHistoryEqualWeightAveragePB, quantileInRecent10YearsEqualWeightAveragePB |
| `stock_a_code_to_symbol` | str | 无需参数 |  |
| `stock_a_congestion_lg` | 964x3 | 无需参数 | date, close, congestion |
| `stock_a_gxl_lg` | 5149x2 | 无需参数 | 日期, 股息率 |
| `stock_a_high_low_statistics` | 500x8 | 无需参数 | date, close, high20, low20, high60, low60, high120, low120 |
| `stock_a_ttm_lyr` | 5144x14 | 无需参数 | date, middlePETTM, averagePETTM, middlePELYR, averagePELYR, quantileInAllHistoryMiddlePeTtm, quantileInRecent10YearsMiddlePeTtm, quantileInAllHistoryAveragePeTtm... (+6) |
| `stock_account_statistics_em` | 101x11 | 无需参数 | 数据日期, 新增投资者-数量, 新增投资者-环比, 新增投资者-同比, 期末投资者-总量, 期末投资者-A股账户, 期末投资者-B股账户, 沪深总市值... (+3) |
| `stock_add_stock` | 1x6 | 无需参数 | 公告日期, 发行方式, 发行价格, 实际公司募集资金总额, 发行费用总额, 实际发行数量 |
| `stock_allotment_cninfo` | 1x57 | 无需参数 | 记录标识, 证券简称, 停牌起始日, 上市公告日期, 配股缴款起始日, 可转配股数量, 停牌截止日, 实际配股数量... (+7) |
| `stock_analyst_detail_em` | 2x9 | 无需参数 | 序号, 股票代码, 股票名称, 调入日期, 最新评级日期, 当前评级名称, 成交价格(前复权), 最新价格... (+1) |
| `stock_balance_sheet_by_report_delisted_em` | 38x319 | 无需参数 | SECUCODE, SECURITY_CODE, SECURITY_NAME_ABBR, ORG_CODE, ORG_TYPE, REPORT_DATE, REPORT_TYPE, REPORT_DATE_NAME... (+7) |
| `stock_board_change_em` | 997x8 | 无需参数 | 板块名称, 涨跌幅, 主力净流入, 板块异动总次数, 板块异动最频繁个股及所属类型-股票代码, 板块异动最频繁个股及所属类型-股票名称, 板块异动最频繁个股及所属类型-买卖方向, 板块具体异动类型列表及出现次数 |
| `stock_board_concept_name_em_async` | 483x12 | 无需参数 | 排名, 板块名称, 板块代码, 最新价, 涨跌额, 涨跌幅, 总市值, 换手率... (+4) |
| `stock_board_industry_index_ths` | 975x7 | 无需参数 | 日期, 开盘价, 最高价, 最低价, 收盘价, 成交量, 成交额 |
| `stock_board_industry_info_ths` | 10x2 | 无需参数 | 项目, 值 |
| `stock_board_industry_name_ths` | 90x2 | 无需参数 | name, code |
| `stock_board_industry_summary_ths` | 90x12 | 无需参数 | 序号, 板块, 涨跌幅, 总成交量, 总成交额, 净流入, 上涨家数, 下跌家数... (+4) |
| `stock_buffett_index_lg` | 5086x6 | 无需参数 | 日期, 收盘价, 总市值, GDP, 近十年分位数, 总历史分位数 |
| `stock_cash_flow_sheet_by_report_delisted_em` | 21x254 | 无需参数 | SECUCODE, SECURITY_CODE, SECURITY_NAME_ABBR, ORG_CODE, ORG_TYPE, REPORT_DATE, REPORT_TYPE, REPORT_DATE_NAME... (+7) |
| `stock_cg_equity_mortgage_cninfo` | 103x10 | 无需参数 | 股票代码, 股票简称, 公告日期, 出质人, 质权人, 质押数量, 占总股本比例, 质押解除数量... (+2) |
| `stock_cg_guarantee_cninfo` | 3105x7 | 无需参数 | 证券代码, 证券简称, 公告统计区间, 担保笔数, 担保金额, 归属于母公司所有者权益, 担保金融占净资产比例 |
| `stock_changes_em` | 1645x5 | symbol="大笔买入" | 时间, 代码, 名称, 板块, 相关信息 |
| `stock_circulate_stock_holder` | 870x7 | 无需参数 | 截止日期, 公告日期, 编号, 股东名称, 持股数量, 占流通股比例, 股本性质 |
| `stock_comment_detail_scrd_desire_em` | 5x6 | 无需参数 | 交易日期, 股票代码, 参与意愿, 5日平均参与意愿, 参与意愿变化, 5日平均变化 |
| `stock_comment_detail_scrd_focus_em` | 30x2 | 无需参数 | 交易日, 用户关注指数 |
| `stock_comment_detail_zhpj_lspf_em` | 30x2 | 无需参数 | 交易日, 评分 |
| `stock_comment_detail_zlkp_jgcyd_em` | 37x2 | 无需参数 | 交易日, 机构参与度 |
| `stock_dividend_cninfo` | 27x11 | 无需参数 | 实施方案公告日期, 分红类型, 送股比例, 转增比例, 派息比例, 股权登记日, 除权日, 派息日... (+3) |
| `stock_dzjy_mrmx` | 3x9 | 无需参数 | 序号, 交易日期, 证券代码, 证券简称, 成交价, 成交量, 成交额, 买方营业部... (+1) |
| `stock_dzjy_mrtj` | 80x12 | 无需参数 | 序号, 交易日期, 证券代码, 证券简称, 涨跌幅, 收盘价, 成交价, 折溢率... (+4) |
| `stock_ebs_lg` | 5083x4 | 无需参数 | 日期, 沪深300指数, 股债利差, 股债利差均线 |
| `stock_esg_rft_sina` | 100x13 | 无需参数 | 股票代码, ESG评分, ESG评分日期, 环境总评, 环境总评日期, 社会责任总评, 社会责任总评日期, 治理总评... (+5) |
| `stock_fhps_detail_em` | 11x19 | 无需参数 | 报告期, 业绩披露日期, 送转股份-送转总比例, 送转股份-送股比例, 送转股份-转股比例, 现金分红-现金分红比例, 现金分红-现金分红比例描述, 现金分红-股息率... (+7) |
| `stock_fhps_detail_ths` | 21x11 | 无需参数 | 报告期, 董事会日期, 股东大会预案公告日期, 实施公告日, 分红方案说明, A股股权登记日, A股除权除息日, 分红总额... (+3) |
| `stock_financial_abstract` | 80x97 | 无需参数 | 选项, 指标, 20250930, 20250630, 20250331, 20241231, 20240930, 20240630... (+7) |
| `stock_financial_abstract_new_ths` | 1200x10 | 无需参数 | report_date, report_name, report_period, quarter_name, metric_name, value, single, yoy... (+2) |
| `stock_financial_abstract_ths` | 109x25 | 无需参数 | 报告期, 净利润, 净利润同比增长率, 扣非净利润, 扣非净利润同比增长率, 营业总收入, 营业总收入同比增长率, 基本每股收益... (+7) |
| `stock_financial_analysis_indicator` | 0x0 | 无需参数 |  |
| `stock_financial_analysis_indicator_em` | 22x140 | 无需参数 | SECUCODE, SECURITY_CODE, SECURITY_NAME_ABBR, ORG_CODE, ORG_TYPE, REPORT_DATE, REPORT_TYPE, REPORT_DATE_NAME... (+7) |
| `stock_financial_benefit_new_ths` | 2550x10 | 无需参数 | report_date, report_name, report_period, quarter_name, metric_name, value, single, yoy... (+2) |
| `stock_financial_benefit_ths` | 109x45 | 无需参数 | 报告期, 报表核心指标, *净利润, *营业总收入, *营业总成本, *归属于母公司所有者的净利润, *扣除非经常性损益后的净利润, 报表全部指标... (+7) |
| `stock_financial_cash_new_ths` | 4500x10 | 无需参数 | report_date, report_name, report_period, quarter_name, metric_name, value, single, yoy... (+2) |
| `stock_financial_cash_ths` | 101x75 | 无需参数 | 报告期, 报表核心指标, *现金及现金等价物净增加额, *经营活动产生的现金流量净额, *投资活动产生的现金流量净额, *筹资活动产生的现金流量净额, *期末现金及现金等价物余额, 报表全部指标... (+7) |
| `stock_financial_debt_new_ths` | 5950x10 | 无需参数 | report_date, report_name, report_period, quarter_name, metric_name, value, single, yoy... (+2) |
| `stock_financial_debt_ths` | 108x81 | 无需参数 | 报告期, 报表核心指标, *所有者权益（或股东权益）合计, *资产合计, *负债合计, *归属于母公司所有者权益合计, 报表全部指标, 流动资产... (+7) |
| `stock_financial_hk_analysis_indicator_em` | 9x36 | 无需参数 | SECUCODE, SECURITY_CODE, SECURITY_NAME_ABBR, ORG_CODE, REPORT_DATE, DATE_TYPE_CODE, PER_NETCASH_OPERATE, PER_OI... (+7) |
| `stock_financial_hk_report_em` | 1124x11 | 无需参数 | SECUCODE, SECURITY_CODE, SECURITY_NAME_ABBR, ORG_CODE, REPORT_DATE, DATE_TYPE_CODE, FISCAL_YEAR, STD_ITEM_CODE... (+3) |
| `stock_financial_report_sina` | 114x147 | 无需参数 | 报告日, 流动资产, 货币资金, 结算备付金, 拆出资金, 交易性金融资产, 买入返售金融资产, 衍生金融资产... (+7) |
| `stock_financial_us_analysis_indicator_em` | 20x49 | 无需参数 | SECUCODE, SECURITY_CODE, SECURITY_NAME_ABBR, ORG_CODE, SECURITY_INNER_CODE, ACCOUNTING_STANDARDS, NOTICE_DATE, START_DATE... (+7) |
| `stock_financial_us_report_em` | 645x9 | 无需参数 | SECUCODE, SECURITY_CODE, SECURITY_NAME_ABBR, REPORT_DATE, REPORT_TYPE, REPORT, STD_ITEM_CODE, AMOUNT... (+1) |
| `stock_fund_flow_concept` | 387x11 | 无需参数 | 序号, 行业, 行业指数, 行业-涨跌幅, 流入资金, 流出资金, 净额, 公司家数... (+3) |
| `stock_fund_flow_industry` | 90x11 | 无需参数 | 序号, 行业, 行业指数, 行业-涨跌幅, 流入资金, 流出资金, 净额, 公司家数... (+3) |
| `stock_fund_stock_holder` | 992x7 | 无需参数 | 基金名称, 基金代码, 持仓数量, 占流通股比例, 持股市值, 占净值比例, 截止日期 |
| `stock_gdfx_free_top_10_em` | 10x8 | 无需参数 | 名次, 股东名称, 股东性质, 股份类型, 持股数, 占总流通股本持股比例, 增减, 变动比率 |
| `stock_gdfx_top_10_em` | 10x7 | 无需参数 | 名次, 股东名称, 股份类型, 持股数, 占总股本持股比例, 增减, 变动比率 |
| `stock_gpzy_industry_data_em` | 127x8 | 无需参数 | 序号, 行业, 平均质押比例, 公司家数, 质押总笔数, 质押总股本, 最新质押市值, 统计时间 |
| `stock_gsrl_gsdt_em` | 70x6 | 无需参数 | 序号, 代码, 简称, 事件类型, 具体事项, 交易日 |
| `stock_history_dividend` | 5675x8 | 无需参数 | 代码, 名称, 上市日期, 累计股息, 年均股息, 分红次数, 融资总额, 融资次数 |
| `stock_history_dividend_detail` | 35x8 | 无需参数 | 公告日期, 送股, 转增, 派息, 进度, 除权除息日, 股权登记日, 红股上市日 |
| `stock_hk_company_profile_em` | 1x17 | 无需参数 | 公司名称, 英文名称, 注册地, 注册地址, 公司成立日期, 所属行业, 董事长, 公司秘书... (+7) |
| `stock_hk_daily` | 5410x6 | 无需参数 | date, open, high, low, close, volume |
| `stock_hk_dividend_payout_em` | 19x7 | 无需参数 | 最新公告日期, 财政年度, 分红方案, 分配类型, 除净日, 截至过户日, 发放日 |
| `stock_hk_famous_spot_em` | 100x12 | 无需参数 | 序号, 代码, 名称, 最新价, 涨跌额, 涨跌幅, 今开, 最高... (+4) |
| `stock_hk_fhpx_detail_ths` | 92x9 | 无需参数 | 公告日期, 方案, 除净日, 派息日, 过户日期起止日-起始, 过户日期起止日-截止, 类型, 进度... (+1) |
| `stock_hk_financial_indicator_em` | 1x21 | 无需参数 | 基本每股收益(元), 每股净资产(元), 法定股本(股), 每手股, 每股股息TTM(港元), 派息比率(%), 已发行股本(股), 已发行股本-H股(股)... (+7) |
| `stock_hk_growth_comparison_em` | 1x10 | 无需参数 | 代码, 简称, 基本每股收益同比增长率, 基本每股收益同比增长率排名, 营业收入同比增长率, 营业收入同比增长率排名, 营业利润率同比增长率, 营业利润率同比增长率排名... (+2) |
| `stock_hk_hist` | 4851x11 | 无需参数 | 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅... (+3) |
| `stock_hk_hist_min_em` | 1655x8 | 无需参数 | 时间, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 最新价 |
| `stock_hk_hot_rank_detail_em` | 120x3 | 无需参数 | 时间, 排名, 证券代码 |
| `stock_hk_hot_rank_latest_em` | 10x2 | 无需参数 | item, value |
| `stock_hk_index_daily_sina` | 2773x6 | 无需参数 | date, open, high, low, close, volume |
| `stock_hk_index_spot_em` | 351x13 | 无需参数 | 序号, 内部编号, 代码, 名称, 最新价, 涨跌额, 涨跌幅, 今开... (+5) |
| `stock_hk_index_spot_sina` | 38x9 | 无需参数 | 代码, 名称, 最新价, 涨跌额, 涨跌幅, 昨收, 今开, 最高... (+1) |
| `stock_hk_profit_forecast_et` | 43x8 | 无需参数 | 财政年度, 纯利/亏损, 每股盈利, 每股派息, 证券商, 评级, 目标价, 更新日期 |
| `stock_hk_scale_comparison_em` | 1x10 | 无需参数 | 代码, 简称, 总市值, 总市值排名, 流通市值, 流通市值排名, 营业总收入, 营业总收入排名... (+2) |
| `stock_hk_security_profile_em` | 1x14 | 无需参数 | 证券代码, 证券简称, 上市日期, 证券类型, 发行价, 发行量(股), 每手股数, 每股面值... (+6) |
| `stock_hk_valuation_baidu` | 366x2 | 无需参数 | date, value |
| `stock_hk_valuation_comparison_em` | 1x18 | 无需参数 | 代码, 简称, 市盈率-TTM, 市盈率-TTM排名, 市盈率-LYR, 市盈率-LYR排名, 市净率-MRQ, 市净率-MRQ排名... (+7) |
| `stock_hold_change_cninfo` | 5570x10 | 无需参数 | 证券代码, 证券简称, 交易市场, 公告日期, 变动日期, 变动原因, 总股本, 已流通股份... (+2) |
| `stock_hold_control_cninfo` | 5518x8 | 无需参数 | 证券代码, 证券简称, 变动日期, 实际控制人名称, 控股数量, 控股比例, 直接控制人名称, 控制类型 |
| `stock_hold_management_detail_cninfo` | 7066x16 | 无需参数 | 证券代码, 证券简称, 截止日期, 公告日期, 高管姓名, 董监高姓名, 董监高职务, 变动人与董监高关系... (+7) |
| `stock_hold_num_cninfo` | 4151x9 | 无需参数 | 证券代码, 证券简称, 变动日期, 本期股东人数, 上期股东人数, 股东人数增幅, 本期人均持股数量, 上期人均持股数量... (+1) |
| `stock_hot_keyword_em` | 8x5 | 无需参数 | 时间, 股票代码, 概念名称, 概念代码, 热度 |
| `stock_hot_rank_detail_em` | 366x5 | 无需参数 | 时间, 排名, 证券代码, 新晋粉丝, 铁杆粉丝 |
| `stock_hot_rank_detail_realtime_em` | 106x2 | 无需参数 | 时间, 排名 |
| `stock_hot_rank_em` | 100x6 | 无需参数 | 当前排名, 代码, 股票名称, 最新价, 涨跌额, 涨跌幅 |
| `stock_hot_rank_latest_em` | 10x2 | 无需参数 | item, value |
| `stock_hot_rank_relate_em` | 10x4 | 无需参数 | 时间, 股票代码, 相关股票代码, 涨跌幅 |
| `stock_hot_search_baidu` | 12x3 | 无需参数 | 名称/代码, 涨跌幅, 综合热度 |
| `stock_hot_up_em` | 100x7 | 无需参数 | 排名较昨日变动, 当前排名, 代码, 股票名称, 最新价, 涨跌额, 涨跌幅 |
| `stock_hsgt_fund_flow_summary_em` | 4x13 | 无需参数 | 交易日, 类型, 板块, 资金方向, 交易状态, 成交净买额, 资金净流入, 当日资金余额... (+5) |
| `stock_hsgt_fund_min_em` | 241x5 | 无需参数 | 日期, 时间, 沪股通, 深股通, 北向资金 |
| `stock_hsgt_stock_statistics_em` | 2x11 | 无需参数 | 持股日期, 股票代码, 股票简称, 当日收盘价, 当日涨跌幅, 持股数量, 持股市值, 持股数量占发行股百分比... (+3) |
| `stock_index_pe_lg` | 5088x8 | 无需参数 | 日期, 指数, 等权静态市盈率, 静态市盈率, 静态市盈率中位数, 等权滚动市盈率, 滚动市盈率, 滚动市盈率中位数 |
| `stock_individual_fund_flow` | 121x13 | 无需参数 | 日期, 收盘价, 涨跌幅, 主力净流入-净额, 主力净流入-净占比, 超大单净流入-净额, 超大单净流入-净占比, 大单净流入-净额... (+5) |
| `stock_individual_info_em` | 9x2 | 无需参数 | item, value |
| `stock_industry_category_cninfo` | 294x8 | 无需参数 | 类目编码, 类目名称, 终止日期, 行业类型, 行业类型编码, 类目名称英文, 父类编码, 分级 |
| `stock_industry_change_cninfo` | 11x11 | 无需参数 | 新证券简称, 行业中类, 行业大类, 行业次类, 行业门类, 机构名称, 行业编码, 分类标准... (+3) |
| `stock_industry_clf_hist_sw` | 12717x4 | 无需参数 | symbol, start_date, industry_code, update_time |
| `stock_info_change_name` | 7x2 | 无需参数 | index, name |
| `stock_info_cjzc_em` | 400x4 | 无需参数 | 标题, 摘要, 发布时间, 链接 |
| `stock_info_global_cls` | 20x4 | 无需参数 | 标题, 内容, 发布日期, 发布时间 |
| `stock_info_global_em` | 200x4 | 无需参数 | 标题, 摘要, 发布时间, 链接 |
| `stock_info_global_futu` | 50x4 | 无需参数 | 标题, 内容, 发布时间, 链接 |
| `stock_info_global_sina` | 20x2 | 无需参数 | 时间, 内容 |
| `stock_info_global_ths` | 20x4 | 无需参数 | 标题, 内容, 发布时间, 链接 |
| `stock_info_sh_delist` | 149x4 | 无需参数 | 公司代码, 公司简称, 上市日期, 暂停上市日期 |
| `stock_info_sz_change_name` | 1734x5 | 无需参数 | 变更日期, 证券代码, 证券简称, 变更前全称, 变更后全称 |
| `stock_info_sz_delist` | 200x4 | 无需参数 | 证券代码, 证券简称, 上市日期, 终止上市日期 |
| `stock_info_sz_name_code` | 2884x7 | symbol="A股列表" | 板块, A股代码, A股简称, A股上市日期, A股总股本, A股流通股本, 所属行业 |
| `stock_inner_trade_xq` | 18423x9 | 无需参数 | 股票代码, 股票名称, 变动日期, 变动人, 变动股数, 成交均价, 变动后持股数, 与董监高关系... (+1) |
| `stock_institute_hold` | 605x8 | 无需参数 | 证券代码, 证券简称, 机构数, 机构数变化, 持股比例, 持股比例增幅, 占流通股比例, 占流通股比例增幅 |
| `stock_institute_hold_detail` | 0x0 | 无需参数 |  |
| `stock_institute_recommend` | 0x8 | 无需参数 | 股票代码, 股票名称, 最新评级, 目标价, 评级日期, 综合评级, 平均涨幅, 行业 |
| `stock_institute_recommend_detail` | 653x8 | 无需参数 | 股票代码, 股票名称, 目标价, 最新评级, 评级机构, 分析师, 行业, 评级日期 |
| `stock_intraday_em` | 4308x4 | 无需参数 | 时间, 成交价, 手数, 买卖盘性质 |
| `stock_ipo_info` | 17x2 | 无需参数 | item, value |
| `stock_ipo_summary_cninfo` | 1x15 | 无需参数 | 股票代码, 招股公告日期, 中签率公告日, 每股面值, 总发行数量, 发行前每股净资产, 摊薄发行市盈率, 募集资金净额... (+7) |
| `stock_irm_ans_cninfo` | 1x7 | 无需参数 | 股票代码, 公司简称, 问题, 回答内容, 提问者, 提问时间, 回答时间 |
| `stock_irm_cninfo` | 355x14 | 无需参数 | 股票代码, 公司简称, 行业, 行业代码, 问题, 提问者, 来源, 提问时间... (+6) |
| `stock_js_weibo_nlp_time` | dict(6) | 无需参数 |  |
| `stock_js_weibo_report` | 50x2 | 无需参数 | name, rate |
| `stock_lh_yyb_control` | 140x6 | 无需参数 | 序号, 营业部名称, 携手营业部家数, 年内最佳携手对象, 年内最佳携手股票数, 年内最佳携手成功率 |
| `stock_lhb_detail_daily_sina` | 55x8 | 无需参数 | 序号, 股票代码, 股票名称, 收盘价, 对应值, 成交量, 成交额, 指标 |
| `stock_lhb_ggtj_sina` | 184x8 | 无需参数 | 股票代码, 股票名称, 上榜次数, 累积购买额, 累积卖出额, 净额, 买入席位数, 卖出席位数 |
| `stock_lhb_jgmmtj_em` | 377x16 | 无需参数 | 序号, 代码, 名称, 收盘价, 涨跌幅, 买方机构数, 卖方机构数, 机构买入总额... (+7) |
| `stock_lhb_jgmx_sina` | 203x6 | 无需参数 | 股票代码, 股票名称, 交易日期, 机构席位买入额, 机构席位卖出额, 类型 |
| `stock_lhb_jgzz_sina` | 125x7 | 无需参数 | 股票代码, 股票名称, 累积买入额, 买入次数, 累积卖出额, 卖出次数, 净额 |
| `stock_lhb_stock_detail_date_em` | 73x3 | 无需参数 | 序号, 股票代码, 交易日 |
| `stock_lhb_stock_detail_em` | 10x8 | 无需参数 | 序号, 交易营业部名称, 买入金额, 买入金额-占总成交比例, 卖出金额, 卖出金额-占总成交比例, 净额, 类型 |
| `stock_main_stock_holder` | 813x10 | 无需参数 | 编号, 股东名称, 持股数量, 持股比例, 股本性质, 截至日期, 公告日期, 股东说明... (+2) |
| `stock_management_change_ths` | 19x7 | 无需参数 | 变动日期, 变动人, 与公司高管关系, 变动数量, 交易均价, 剩余股数, 股份变动途径 |
| `stock_margin_detail_sse` | 1740x9 | 无需参数 | 信用交易日期, 标的证券代码, 标的证券简称, 融资余额, 融资买入额, 融资偿还额, 融券余量, 融券卖出量... (+1) |
| `stock_margin_detail_szse` | 1811x8 | 无需参数 | 证券代码, 证券简称, 融资买入额, 融资余额, 融券卖出量, 融券余量, 融券余额, 融资融券余额 |
| `stock_margin_ratio_pa` | 2055x4 | 无需参数 | 证券代码, 证券简称, 融资比例, 融券比例 |
| `stock_margin_sse` | 7x7 | start_date="20250101", end_date="20250110" | 信用交易日期, 融资余额, 融资买入额, 融券余量, 融券余量金额, 融券卖出量, 融资融券余额 |
| `stock_margin_underlying_info_szse` | 1681x8 | 无需参数 | 证券代码, 证券简称, 融资标的, 融券标的, 当日可融资, 当日可融券, 融券卖出价格限制, 涨跌幅限制 |
| `stock_market_activity_legu` | 12x2 | 无需参数 | item, value |
| `stock_market_fund_flow` | 121x15 | 无需参数 | 日期, 上证-收盘价, 上证-涨跌幅, 深证-收盘价, 深证-涨跌幅, 主力净流入-净额, 主力净流入-净占比, 超大单净流入-净额... (+7) |
| `stock_market_pb_lg` | 5149x5 | 无需参数 | 日期, 指数, 市净率, 等权市净率, 市净率中位数 |
| `stock_market_pe_lg` | 340x3 | 无需参数 | 日期, 指数, 平均市盈率 |
| `stock_new_ipo_cninfo` | 505x13 | 无需参数 | 证劵代码, 证券简称, 上市日期, 申购日期, 发行价, 总发行数量, 发行市盈率, 上网发行中签率... (+5) |
| `stock_profile_cninfo` | 1x26 | 无需参数 | 公司名称, 英文名称, 曾用简称, A股代码, A股简称, B股代码, B股简称, H股代码... (+7) |
| `stock_profit_forecast_ths` | 3x6 | 无需参数 | 年度, 预测机构数, 最小值, 均值, 最大值, 行业平均数 |
| `stock_profit_sheet_by_report_delisted_em` | 39x203 | 无需参数 | SECUCODE, SECURITY_CODE, SECURITY_NAME_ABBR, ORG_CODE, ORG_TYPE, REPORT_DATE, REPORT_TYPE, REPORT_DATE_NAME... (+7) |
| `stock_qsjy_em` | 38x14 | 无需参数 | 简称, 代码, 当月净利润-净利润, 当月净利润-同比增长, 当月净利润-环比增长, 当年累计净利润-累计净利润, 当年累计净利润-同比增长, 当月营业收入-营业收入... (+6) |
| `stock_rank_cxfl_ths` | 279x10 | 无需参数 | 序号, 股票代码, 股票简称, 涨跌幅, 最新价, 成交量, 基准日成交量, 放量天数... (+2) |
| `stock_rank_cxg_ths` | 97x8 | 无需参数 | 序号, 股票代码, 股票简称, 涨跌幅, 换手率, 最新价, 前期高点, 前期高点日期 |
| `stock_rank_cxsl_ths` | 333x10 | 无需参数 | 序号, 股票代码, 股票简称, 涨跌幅, 最新价, 成交量, 基准日成交量, 缩量天数... (+2) |
| `stock_rank_forecast_cninfo` | 391x11 | 无需参数 | 证券代码, 证券简称, 发布日期, 研究机构简称, 研究员名称, 投资评级, 是否首次评级, 评级变化... (+3) |
| `stock_rank_ljqs_ths` | 532x8 | 无需参数 | 序号, 股票代码, 股票简称, 最新价, 量价齐升天数, 阶段涨幅, 累计换手率, 所属行业 |
| `stock_rank_lxsz_ths` | 87x10 | 无需参数 | 序号, 股票代码, 股票简称, 收盘价, 最高价, 最低价, 连涨天数, 连续涨跌幅... (+2) |
| `stock_rank_xzjp_ths` | 16x12 | 无需参数 | 序号, 举牌公告日, 股票代码, 股票简称, 现价, 涨跌幅, 举牌方, 增持数量... (+4) |
| `stock_report_disclosure` | 4603x7 | 无需参数 | 股票代码, 股票简称, 首次预约, 初次变更, 二次变更, 三次变更, 实际披露 |
| `stock_report_fund_hold_detail` | 10x7 | 无需参数 | 序号, 股票代码, 股票简称, 持股数, 持股市值, 占总股本比例, 占流通股本比例 |
| `stock_research_report_em` | 224x16 | 无需参数 | 序号, 股票代码, 股票简称, 报告名称, 东财评级, 机构, 近一月个股研报数, 2025-盈利预测-收益... (+7) |
| `stock_restricted_release_queue_em` | 4x13 | 无需参数 | 序号, 解禁时间, 解禁股东数, 解禁数量, 实际解禁数量, 未解禁数量, 实际解禁数量市值, 占总市值比例... (+5) |
| `stock_restricted_release_queue_sina` | 8x7 | 无需参数 | 代码, 名称, 解禁日期, 解禁数量, 解禁股流通市值, 上市批次, 公告日期 |
| `stock_restricted_release_stockholder_em` | 2x9 | 无需参数 | 序号, 股东名称, 解禁数量, 实际解禁数量, 解禁市值, 锁定期, 剩余未解禁数量, 限售股类型... (+1) |
| `stock_restricted_release_summary_em` | 29x8 | 无需参数 | 序号, 解禁时间, 当日解禁股票家数, 解禁数量, 实际解禁数量, 实际解禁市值, 沪深300指数, 沪深300指数涨跌幅 |
| `stock_sector_detail` | 167x20 | 无需参数 | symbol, code, name, trade, pricechange, changepercent, buy, sell... (+7) |
| `stock_sector_spot` | 49x13 | 无需参数 | label, 板块, 公司家数, 平均价格, 涨跌额, 涨跌幅, 总成交量, 总成交额... (+5) |
| `stock_sgt_reference_exchange_rate_sse` | 2000x4 | 无需参数 | 适用日期, 参考汇率买入价, 参考汇率卖出价, 货币种类 |
| `stock_sgt_reference_exchange_rate_szse` | 2194x4 | 无需参数 | 适用日期, 参考汇率买入价, 参考汇率卖出价, 货币种类 |
| `stock_sgt_settlement_exchange_rate_sse` | 2000x4 | 无需参数 | 适用日期, 买入结算汇兑比率, 卖出结算汇兑比率, 货币种类 |
| `stock_sgt_settlement_exchange_rate_szse` | 2186x4 | 无需参数 | 适用日期, 买入结算汇兑比率, 卖出结算汇兑比率, 货币种类 |
| `stock_share_change_cninfo` | 41x44 | 无需参数 | 证券简称, 机构名称, 境外法人持股, 证券投资基金持股, 国家持股-受限, 国有法人持股, 配售法人股, 发起人股份... (+7) |
| `stock_share_hold_change_bse` | 8x10 | 无需参数 | 代码, 简称, 姓名, 职务, 变动日期, 变动股数, 变动前持股数, 变动后持股数... (+2) |
| `stock_share_hold_change_sse` | 27x13 | 无需参数 | 公司代码, 公司名称, 姓名, 职务, 股票种类, 货币种类, 本次变动前持股数, 变动数... (+5) |
| `stock_shareholder_change_ths` | 10x7 | 无需参数 | 公告日期, 变动股东, 变动数量, 交易均价, 剩余股份总数, 变动期间, 变动途径 |
| `stock_sse_deal_daily` | 8x6 | 无需参数 | 单日情况, 股票, 主板A, 主板B, 科创板, 股票回购 |
| `stock_sse_summary` | 8x4 | 无需参数 | 项目, 股票, 主板, 科创板 |
| `stock_sy_profile_em` | 17x8 | 无需参数 | 报告期, 商誉, 商誉减值, 净资产, 商誉占净资产比例, 商誉减值占净资产比例, 净利润规模, 商誉减值占净利润比例 |
| `stock_szse_area_summary` | 34x7 | 无需参数 | 序号, 地区, 总交易额, 占市场, 股票交易额, 基金交易额, 债券交易额 |
| `stock_szse_sector_summary` | 20x9 | 无需参数 | 项目名称, 项目名称-英文, 交易天数, 成交金额-人民币元, 成交金额-占总计, 成交股数-股数, 成交股数-占总计, 成交笔数-笔... (+1) |
| `stock_szse_summary` | 14x5 | 无需参数 | 证券类别, 数量, 成交金额, 总市值, 流通市值 |
| `stock_us_daily` | 2671x6 | 无需参数 | date, open, high, low, close, volume |
| `stock_us_hist` | 10084x11 | 无需参数 | 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅... (+3) |
| `stock_us_hist_min_em` | 1955x8 | 无需参数 | 时间, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 最新价 |
| `stock_us_valuation_baidu` | 251x2 | 无需参数 | date, value |
| `stock_yjkb_em` | 1657x16 | 无需参数 | 序号, 股票代码, 股票简称, 每股收益, 营业收入-营业收入, 营业收入-去年同期, 营业收入-同比增长, 营业收入-季度环比增长... (+7) |
| `stock_yjyg_em` | 2387x11 | 无需参数 | 序号, 股票代码, 股票简称, 预测指标, 业绩变动, 预测数值, 业绩变动幅度, 业绩变动原因... (+3) |
| `stock_zh_a_cdr_daily` | 1307x10 | 无需参数 | date, prevclose, open, high, low, close, volume, amount... (+2) |
| `stock_zh_a_daily` | 7x9 | symbol="sz000001", start_date="20250101", end_date="20250110", adjust="qfq" | date, open, high, low, close, volume, amount, outstanding_share... (+1) |
| `stock_zh_a_disclosure_report_cninfo` | 38x5 | 无需参数 | 代码, 简称, 公告标题, 公告时间, 公告链接 |
| `stock_zh_a_gbjg_em` | 13x9 | 无需参数 | 变更日期, 总股本, 流通受限股份, 其他内资持股(受限), 境内法人持股(受限), 境内自然人持股(受限), 已流通股份, 已上市流通A股... (+1) |
| `stock_zh_a_gdhs_detail_em` | 64x15 | 无需参数 | 股东户数统计截止日, 区间涨跌幅, 股东户数-本次, 股东户数-上次, 股东户数-增减, 股东户数-增减比例, 户均持股市值, 户均持股数量... (+7) |
| `stock_zh_a_hist_pre_min_em` | 256x8 | 无需参数 | 时间, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 最新价 |
| `stock_zh_a_minute` | 1970x6 | symbol="sz000001", period="5" | day, open, high, low, close, volume |
| `stock_zh_a_new` | 118x10 | 无需参数 | symbol, code, name, open, high, low, volume, amount... (+2) |
| `stock_zh_b_daily` | 7811x8 | 无需参数 | date, open, high, low, close, volume, outstanding_share, turnover |
| `stock_zh_b_minute` | 1970x6 | 无需参数 | day, open, high, low, close, volume |
| `stock_zh_b_spot` | 79x13 | 无需参数 | 代码, 名称, 最新价, 涨跌额, 涨跌幅, 买入, 卖出, 昨收... (+5) |
| `stock_zh_dupont_comparison_em` | 8x19 | 无需参数 | 代码, 简称, ROE-3年平均, ROE-22A, ROE-23A, ROE-24A, 净利率-3年平均, 净利率-22A... (+7) |
| `stock_zh_growth_comparison_em` | 8x21 | 无需参数 | 代码, 简称, 基本每股收益增长率-3年复合, 基本每股收益增长率-24A, 基本每股收益增长率-TTM, 基本每股收益增长率-25E, 基本每股收益增长率-26E, 基本每股收益增长率-27E... (+7) |
| `stock_zh_index_daily` | 8604x6 | symbol="sh000001" | date, open, high, low, close, volume |
| `stock_zh_index_daily_tx` | 689x6 | 无需参数 | date, open, close, high, low, amount |
| `stock_zh_index_hist_csindex` | 1464x16 | 无需参数 | 日期, 指数代码, 指数中文全称, 指数中文简称, 指数英文全称, 指数英文简称, 开盘, 最高... (+7) |
| `stock_zh_index_value_csindex` | 20x10 | 无需参数 | 日期, 指数代码, 指数中文全称, 指数中文简称, 指数英文全称, 指数英文简称, 市盈率1, 市盈率2... (+2) |
| `stock_zh_kcb_daily` | 1523x10 | 无需参数 | date, open, high, low, close, volume, after_volume, after_amount... (+2) |
| `stock_zh_kcb_spot` | 604x19 | 无需参数 | 代码, 名称, 最新价, 涨跌额, 涨跌幅, 买入, 卖出, 昨收... (+7) |
| `stock_zh_scale_comparison_em` | 1x10 | 无需参数 | 代码, 简称, 总市值, 总市值排名, 流通市值, 流通市值排名, 营业收入, 营业收入排名... (+2) |
| `stock_zh_valuation_baidu` | 366x2 | 无需参数 | date, value |
| `stock_zh_valuation_comparison_em` | 8x20 | 无需参数 | 排名, 代码, 简称, PEG, 市盈率-TTM, 市盈率-25E, 市盈率-26E, 市盈率-27E... (+7) |
| `stock_zh_vote_baidu` | 4x5 | 无需参数 | 周期, 看涨, 看跌, 看涨比例, 看跌比例 |
| `stock_zt_pool_em` | 0x0 | date="20250110" |  |
| `stock_zt_pool_previous_em` | 0x0 | date="20250110" |  |
| `stock_zt_pool_strong_em` | 0x0 | date="20250110" |  |
| `stock_zt_pool_sub_new_em` | 0x0 | date="20250110" |  |
| `stock_zygc_em` | 73x11 | 无需参数 | 股票代码, 报告日期, 分类类型, 主营构成, 主营收入, 收入比例, 主营成本, 成本比例... (+3) |
| `stock_zyjs_ths` | 1x5 | 无需参数 | 股票代码, 主营业务, 产品类型, 产品名称, 经营范围 |

## 宏观数据

✅ 108 成功 | ❌ 3 失败 | ⏱ 113 超时 | ⊘ 2 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `macro_australia_bank_rate` | 194x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_australia_cpi_quarterly` | 73x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_australia_cpi_yearly` | 73x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_australia_ppi_quarterly` | 73x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_australia_retail_rate_monthly` | 210x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_australia_trade` | 188x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_australia_unemployment_rate` | 219x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_bank_brazil_interest_rate` | 156x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_bank_china_interest_rate` | 218x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_bank_newzealand_interest_rate` | 240x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_bank_switzerland_interest_rate` | 75x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_canada_bank_rate` | 175x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_canada_core_cpi_yearly` | 217x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_canada_cpi_yearly` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_canada_new_house_rate` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_canada_retail_rate_monthly` | 73x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_canada_unemployment_rate` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_china_bank_financing` | 266x8 | 无需参数 | 日期, 最新值, 涨跌幅, 近3月涨跌幅, 近6月涨跌幅, 近1年涨跌幅, 近2年涨跌幅, 近3年涨跌幅 |
| `macro_china_consumer_goods_retail` | 203x6 | 无需参数 | 月份, 当月, 同比增长, 环比增长, 累计, 累计-同比增长 |
| `macro_china_cpi` | 218x13 | 无需参数 | 月份, 全国-当月, 全国-同比增长, 全国-环比增长, 全国-累计, 城市-当月, 城市-同比增长, 城市-环比增长... (+5) |
| `macro_china_cx_pmi_yearly` | 219x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_china_cx_services_pmi_yearly` | 166x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_china_czsr` | 210x6 | 无需参数 | 月份, 当月, 当月-同比增长, 当月-环比增长, 累计, 累计-同比增长 |
| `macro_china_daily_energy` | 1207x4 | 无需参数 | 日期, 沿海六大电库存, 日耗, 存煤可用天数 |
| `macro_china_fdi` | 185x6 | 无需参数 | 月份, 当月, 当月-同比增长, 当月-环比增长, 累计, 累计-同比增长 |
| `macro_china_freight_index` | 4994x8 | 无需参数 | 截止日期, 波罗的海好望角型船运价指数BCI, 灵便型船综合运价指数BHMI, 波罗的海超级大灵便型船BSI指数, 波罗的海综合运价指数BDI, HRCI国际集装箱租船指数, 油轮运价指数成品油运价指数BCTI, 油轮运价指数原油运价指数BDTI |
| `macro_china_fx_reserves_yearly` | 132x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_china_gdp` | 80x9 | 无需参数 | 季度, 国内生产总值-绝对值, 国内生产总值-同比增长, 第一产业-绝对值, 第一产业-同比增长, 第二产业-绝对值, 第二产业-同比增长, 第三产业-绝对值... (+1) |
| `macro_china_gdp_yearly` | 61x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_china_gdzctz` | 199x5 | 无需参数 | 月份, 当月, 同比增长, 环比增长, 自年初累计 |
| `macro_china_gyzjz` | 199x4 | 无需参数 | 月份, 同比增长, 累计增长, 发布时间 |
| `macro_china_hgjck` | 218x11 | 无需参数 | 月份, 当月出口额-金额, 当月出口额-同比增长, 当月出口额-环比增长, 当月进口额-金额, 当月进口额-同比增长, 当月进口额-环比增长, 累计出口额-金额... (+3) |
| `macro_china_hk_building_volume` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_china_hk_cpi_ratio` | 172x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_china_hk_gbp` | 73x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_china_hk_gbp_ratio` | 73x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_china_hk_market_info` | 2211x17 | 无需参数 | 日期, 1W-定价, 1W-涨跌幅, 2W-定价, 2W-涨跌幅, 1M-定价, 1M-涨跌幅, 3M-定价... (+7) |
| `macro_china_hk_ppi` | 73x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_china_hk_rate_of_unemployment` | 219x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_china_hk_trade_diff_ratio` | 217x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_china_insurance_income` | 257x8 | 无需参数 | 日期, 最新值, 涨跌幅, 近3月涨跌幅, 近6月涨跌幅, 近1年涨跌幅, 近2年涨跌幅, 近3年涨跌幅 |
| `macro_china_international_tourism_fx` | 264x4 | 无需参数 | 统计年度, 指标, 数量, 比重 |
| `macro_china_market_margin_sh` | 3873x7 | 无需参数 | 日期, 融资买入额, 融资余额, 融券卖出量, 融券余量, 融券余额, 融资融券余额 |
| `macro_china_market_margin_sz` | 3675x7 | 无需参数 | 日期, 融资买入额, 融资余额, 融券卖出量, 融券余量, 融券余额, 融资融券余额 |
| `macro_china_mobile_number` | 169x8 | 无需参数 | 日期, 最新值, 涨跌幅, 近3月涨跌幅, 近6月涨跌幅, 近1年涨跌幅, 近2年涨跌幅, 近3年涨跌幅 |
| `macro_china_national_tax_receipts` | 82x4 | 无需参数 | 季度, 税收收入合计, 较上年同期, 季度环比 |
| `macro_china_new_financial_credit` | 218x6 | 无需参数 | 月份, 当月, 当月-同比增长, 当月-环比增长, 累计, 累计-同比增长 |
| `macro_china_new_house_price` | 364x8 | 无需参数 | 日期, 城市, 新建商品住宅价格指数-同比, 新建商品住宅价格指数-环比, 新建商品住宅价格指数-定基, 二手住宅价格指数-同比, 二手住宅价格指数-环比, 二手住宅价格指数-定基 |
| `macro_china_passenger_load_factor` | 238x3 | 无需参数 | 统计时间, 客座率, 载运率 |
| `macro_china_pmi` | 218x5 | 无需参数 | 月份, 制造业-指数, 制造业-同比增长, 非制造业-指数, 非制造业-同比增长 |
| `macro_china_ppi` | 242x4 | 无需参数 | 月份, 当月, 当月同比增长, 累计 |
| `macro_china_qyspjg` | 253x13 | 无需参数 | 月份, 总指数-指数值, 总指数-同比增长, 总指数-环比增长, 农产品-指数值, 农产品-同比增长, 农产品-环比增长, 矿产品-指数值... (+5) |
| `macro_china_reserve_requirement_ratio` | 58x11 | 无需参数 | 公布时间, 生效时间, 大型金融机构-调整前, 大型金融机构-调整后, 大型金融机构-调整幅度, 中小金融机构-调整前, 中小金融机构-调整后, 中小金融机构-调整幅度... (+3) |
| `macro_china_rmb` | 791x49 | 无需参数 | 日期, 美元/人民币_中间价, 美元/人民币_涨跌幅, 欧元/人民币_中间价, 欧元/人民币_涨跌幅, 100日元/人民币_中间价, 100日元/人民币_涨跌幅, 港元/人民币_中间价... (+7) |
| `macro_china_society_electricity` | 236x17 | 无需参数 | 统计时间, 全社会用电量, 全社会用电量同比, 各行业用电量合计, 各行业用电量合计同比, 第一产业用电量, 第一产业用电量同比, 第二产业用电量... (+7) |
| `macro_china_stock_market_cap` | 219x13 | 无需参数 | 数据日期, 发行总股本-上海, 发行总股本-深圳, 市价总值-上海, 市价总值-深圳, 成交金额-上海, 成交金额-深圳, 成交量-上海... (+5) |
| `macro_china_wbck` | 218x5 | 无需参数 | 月份, 当月, 同比增长, 环比增长, 累计 |
| `macro_china_whxd` | 218x5 | 无需参数 | 月份, 当月, 同比增长, 环比增长, 累计 |
| `macro_china_xfzxx` | 229x10 | 无需参数 | 月份, 消费者信心指数-指数值, 消费者信心指数-同比增长, 消费者信心指数-环比增长, 消费者满意指数-指数值, 消费者满意指数-同比增长, 消费者满意指数-环比增长, 消费者预期指数-指数值... (+2) |
| `macro_cnbs` | 80x9 | 无需参数 | 年份, 居民部门, 非金融企业部门, 政府部门, 中央政府, 地方政府, 实体经济部门, 金融部门资产方... (+1) |
| `macro_euro_current_account_mom` | 214x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_euro_employment_change_qoq` | 100x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_euro_lme_holding` | 756x19 | 无需参数 | 日期, 铜-多头仓位, 铜-空头仓位, 铜-净仓位, 锌-多头仓位, 锌-空头仓位, 锌-净仓位, 镍-多头仓位... (+7) |
| `macro_euro_lme_stock` | 2636x19 | 无需参数 | 日期, 铜-库存, 铜-注册仓单, 铜-注销仓单, 锡-库存, 锡-注册仓单, 锡-注销仓单, 铅-库存... (+7) |
| `macro_euro_zew_economic_sentiment` | 215x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_germany_cpi_yearly` | 219x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_germany_gdp` | 72x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_germany_ifo` | 219x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_germany_retail_sale_monthly` | 219x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_germany_retail_sale_yearly` | 18x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_germany_trade_adjusted` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_germany_zew` | 219x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_japan_bank_rate` | 197x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_japan_core_cpi_yearly` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_japan_cpi_yearly` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_japan_unemployment_rate` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_rmb_deposit` | 30x13 | 无需参数 | 月份, 新增存款-数量, 新增存款-同比, 新增存款-环比, 新增企业存款-数量, 新增企业存款-同比, 新增企业存款-环比, 新增储蓄存款-数量... (+5) |
| `macro_rmb_loan` | 30x6 | 无需参数 | 月份, 新增人民币贷款-总额, 新增人民币贷款-同比, 新增人民币贷款-环比, 累计人民币贷款-总额, 累计人民币贷款-同比 |
| `macro_stock_finance` | 30x5 | 无需参数 | 月份, 募集资金, 首发募集资金, 增发募集资金, 配股募集资金 |
| `macro_swiss_cpi_yearly` | 19x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_swiss_gbd_bank_rate` | 74x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_swiss_gbd_yearly` | 59x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_swiss_gdp_quarterly` | 74x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_swiss_svme` | 220x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_swiss_trade` | 73x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_bank_rate` | 184x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_core_cpi_monthly` | 215x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_core_cpi_yearly` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_gdp_quarterly` | 70x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_gdp_yearly` | 17x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_halifax_monthly` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_halifax_yearly` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_retail_monthly` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_rightmove_monthly` | 196x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_rightmove_yearly` | 196x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_uk_unemployment_rate` | 218x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_usa_cftc_c_holding` | 1909x37 | 无需参数 | 日期, 纽约原油-多头仓位, 纽约原油-空头仓位, 纽约原油-净仓位, 原糖-多头仓位, 原糖-空头仓位, 原糖-净仓位, 大豆-多头仓位... (+7) |
| `macro_usa_cftc_merchant_currency_holding` | 1909x28 | 无需参数 | 日期, 美元-多头仓位, 美元-空头仓位, 美元-净仓位, 瑞郎-多头仓位, 瑞郎-空头仓位, 瑞郎-净仓位, 纽元-多头仓位... (+7) |
| `macro_usa_cftc_merchant_goods_holding` | 1909x37 | 无需参数 | 日期, 纽约原油-多头仓位, 纽约原油-空头仓位, 纽约原油-净仓位, 原糖-多头仓位, 原糖-空头仓位, 原糖-净仓位, 大豆-多头仓位... (+7) |
| `macro_usa_cftc_nc_holding` | 1909x28 | 无需参数 | 日期, 美元-多头仓位, 美元-空头仓位, 美元-净仓位, 瑞郎-多头仓位, 瑞郎-空头仓位, 瑞郎-净仓位, 纽元-多头仓位... (+7) |
| `macro_usa_cpi_yoy` | 218x4 | 无需参数 | 时间, 发布日期, 现值, 前值 |
| `macro_usa_crude_inner` | 2251x7 | 无需参数 | 日期, 美国国内原油总量-产量, 美国国内原油总量-变化, 美国本土48州原油产量-产量, 美国本土48州原油产量-变化, 美国阿拉斯加州原油产量-产量, 美国阿拉斯加州原油产量-变化 |
| `macro_usa_current_account` | 71x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_usa_gdp_monthly` | 210x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_usa_lmci` | 34x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_usa_phs` | 211x4 | 无需参数 | 时间, 前值, 现值, 发布日期 |
| `macro_usa_real_consumer_spending` | 146x5 | 无需参数 | 商品, 日期, 今值, 预测值, 前值 |
| `macro_usa_rig_count` | 1896x9 | 无需参数 | 日期, 钻井总数_钻井数, 钻井总数_变化, 美国石油钻井_钻井数, 美国石油钻井_变化, 混合钻井_钻井数, 混合钻井_变化, 美国天然气钻井_钻井数... (+1) |

## 指数数据

✅ 62 成功 | ❌ 5 失败 | ⏱ 12 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `index_ai_cx` | 1261x3 | 无需参数 | 日期, AI策略指数, 变化幅度 |
| `index_all_cni` | 1393x10 | 无需参数 | 指数代码, 指数简称, 样本数, 收盘点位, 涨跌幅, PE滚动, 成交量, 成交额... (+2) |
| `index_analysis_daily_sw` | 9x14 | 无需参数 | 指数代码, 指数名称, 发布日期, 收盘指数, 成交量, 涨跌幅, 换手率, 市盈率... (+6) |
| `index_analysis_monthly_sw` | 9x14 | 无需参数 | 指数代码, 指数名称, 发布日期, 收盘指数, 成交量, 涨跌幅, 换手率, 市盈率... (+6) |
| `index_analysis_week_month_sw` | 314x1 | 无需参数 | date |
| `index_analysis_weekly_sw` | 9x14 | 无需参数 | 指数代码, 指数名称, 发布日期, 收盘指数, 成交量, 涨跌幅, 换手率, 市盈率... (+6) |
| `index_awpr_cx` | 127x3 | 无需参数 | 日期, 新经济入职工资溢价水平, 变化值 |
| `index_bei_cx` | 2239x3 | 无需参数 | 日期, 基石经济指数, 变化幅度 |
| `index_bi_cx` | 110x3 | 无需参数 | 日期, 基础指数, 变化值 |
| `index_cci_cx` | 4173x3 | 无需参数 | 日期, 大宗商品指数, 变化值 |
| `index_ci_cx` | 127x3 | 无需参数 | 日期, 资本投入指数, 变化值 |
| `index_component_sw` | 55x5 | 无需参数 | 序号, 证券代码, 证券名称, 最新权重, 计入日期 |
| `index_csindex_all` | 2291x17 | 无需参数 | 指数代码, 指数简称, 指数全称, 基日, 基点, 指数系列, 样本数量, 最新收盘... (+7) |
| `index_dei_cx` | 110x3 | 无需参数 | 日期, 数字经济指数, 变化值 |
| `index_detail_cni` | 500x6 | 无需参数 | 日期, 样本代码, 样本简称, 所属行业, 总市值, 权重 |
| `index_detail_hist_cni` | 500x6 | 无需参数 | 日期, 样本代码, 样本简称, 所属行业, 总市值, 权重 |
| `index_fi_cx` | 110x3 | 无需参数 | 日期, 融合指数, 变化值 |
| `index_global_name_table` | 20x2 | 无需参数 | 指数名称, 代码 |
| `index_hist_cni` | 242x8 | 无需参数 | 日期, 开盘价, 最高价, 最低价, 收盘价, 涨跌幅, 成交量, 成交额 |
| `index_hist_fund_sw` | 4908x6 | 无需参数 | 日期, 收盘指数, 开盘指数, 最高指数, 最低指数, 涨跌幅 |
| `index_hist_sw` | 6333x8 | 无需参数 | 代码, 日期, 收盘, 开盘, 最高, 最低, 成交量, 成交额 |
| `index_hog_spot_price` | 562x8 | 无需参数 | 日期, 指数, 4个月均线, 6个月均线, 12个月均线, 预售均价, 成交均价, 成交均重 |
| `index_ii_cx` | 110x3 | 无需参数 | 日期, 产业指数, 变化值 |
| `index_inner_quote_sugar_msweet` | 3567x13 | 无需参数 | 日期, 利润空间, 泰国糖, 泰国MA5, 巴西MA5, 利润MA5, 巴西MA10, 巴西糖... (+5) |
| `index_kq_fashion` | 41x4 | 无需参数 | 日期, 指数, 涨跌值, 涨跌幅 |
| `index_li_cx` | 127x3 | 无需参数 | 日期, 劳动力投入指数, 变化值 |
| `index_min_sw` | 1440x5 | 无需参数 | 代码, 名称, 价格, 日期, 时间 |
| `index_neaw_cx` | 127x3 | 无需参数 | 日期, 新经济行业入职平均工资水平, 变化值 |
| `index_neei_cx` | 18x3 | 无需参数 | 日期, 新动能指数, 变化幅度 |
| `index_nei_cx` | 118x3 | 无需参数 | 日期, 中国新经济指数, 变化值 |
| `index_option_1000index_min_qvix` | 239x2 | 无需参数 | time, qvix |
| `index_option_1000index_qvix` | 2688x5 | 无需参数 | date, open, high, low, close |
| `index_option_100etf_min_qvix` | 239x2 | 无需参数 | time, qvix |
| `index_option_100etf_qvix` | 2688x5 | 无需参数 | date, open, high, low, close |
| `index_option_300etf_min_qvix` | 239x2 | 无需参数 | time, qvix |
| `index_option_300etf_qvix` | 2688x5 | 无需参数 | date, open, high, low, close |
| `index_option_300index_min_qvix` | 239x2 | 无需参数 | time, qvix |
| `index_option_300index_qvix` | 2688x5 | 无需参数 | date, open, high, low, close |
| `index_option_500etf_min_qvix` | 239x2 | 无需参数 | time, qvix |
| `index_option_500etf_qvix` | 2688x5 | 无需参数 | date, open, high, low, close |
| `index_option_50etf_min_qvix` | 239x2 | 无需参数 | time, qvix |
| `index_option_50index_min_qvix` | 239x2 | 无需参数 | time, qvix |
| `index_option_50index_qvix` | 2688x5 | 无需参数 | date, open, high, low, close |
| `index_option_cyb_min_qvix` | 239x2 | 无需参数 | time, qvix |
| `index_option_cyb_qvix` | 2688x5 | 无需参数 | date, open, high, low, close |
| `index_option_kcb_min_qvix` | 239x2 | 无需参数 | time, qvix |
| `index_option_kcb_qvix` | 2688x5 | 无需参数 | date, open, high, low, close |
| `index_outer_quote_sugar_msweet` | 1780x6 | 无需参数 | 日期, 巴西糖进口成本, 泰国糖进口利润空间, 巴西糖进口利润空间, 泰国糖进口成本, 日照现货价 |
| `index_pmi_com_cx` | 143x3 | 无需参数 | 日期, 综合PMI, 变化值 |
| `index_pmi_man_cx` | 143x3 | 无需参数 | 日期, 制造业PMI, 变化值 |
| `index_pmi_ser_cx` | 143x3 | 无需参数 | 日期, 服务业PMI, 变化值 |
| `index_price_cflp` | 531x4 | 无需参数 | 日期, 定基指数, 环比指数, 同比指数 |
| `index_qli_cx` | 1913x3 | 无需参数 | 日期, 高质量因子指数, 变化幅度 |
| `index_realtime_fund_sw` | 7x5 | 无需参数 | 指数代码, 指数名称, 昨收盘, 日涨跌幅, 年涨跌幅 |
| `index_si_cx` | 110x3 | 无需参数 | 日期, 溢出指数, 变化值 |
| `index_stock_cons` | 300x3 | symbol="000300" | 品种代码, 品种名称, 纳入日期 |
| `index_stock_cons_csindex` | 300x9 | 无需参数 | 日期, 指数代码, 指数名称, 指数英文名称, 成分券代码, 成分券名称, 成分券英文名称, 交易所... (+1) |
| `index_stock_cons_sina` | 300x20 | 无需参数 | symbol, code, name, trade, pricechange, changepercent, buy, sell... (+7) |
| `index_sugar_msweet` | 4881x4 | 无需参数 | 日期, 综合价格, 原糖价格, 现货价格 |
| `index_us_stock_sina` | 5591x7 | 无需参数 | date, open, high, low, close, volume, amount |
| `index_volume_cflp` | 55x4 | 无需参数 | 日期, 定基指数, 环比指数, 同比指数 |
| `index_yw` | 11x5 | 无需参数 | 期数, 景气指数, 规模指数, 效益指数, 市场信心指数 |

## 基金数据

✅ 48 成功 | ❌ 3 失败 | ⏱ 17 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `fund_announcement_dividend_em` | 15x5 | 无需参数 | 基金代码, 公告标题, 基金名称, 公告日期, 报告ID |
| `fund_announcement_personnel_em` | 14x5 | 无需参数 | 基金代码, 公告标题, 基金名称, 公告日期, 报告ID |
| `fund_announcement_report_em` | 100x5 | 无需参数 | 基金代码, 公告标题, 基金名称, 公告日期, 报告ID |
| `fund_aum_em` | 215x7 | 无需参数 | 序号, 基金公司, 成立时间, 全部管理规模, 全部基金数, 全部经理数, 更新日期 |
| `fund_aum_hist_em` | 202x9 | 无需参数 | 序号, 基金公司, 总规模, 股票型, 混合型, 债券型, 指数型, QDII... (+1) |
| `fund_aum_trend_em` | 21x2 | 无需参数 | date, value |
| `fund_balance_position_lg` | 414x3 | 无需参数 | date, close, position |
| `fund_cf_em` | 96x6 | 无需参数 | 序号, 基金代码, 基金简称, 拆分折算日, 拆分类型, 拆分折算 |
| `fund_etf_category_sina` | 382x13 | 无需参数 | 代码, 名称, 最新价, 涨跌额, 涨跌幅, 买入, 卖出, 昨收... (+5) |
| `fund_etf_dividend_sina` | 18x2 | 无需参数 | 日期, 累计分红 |
| `fund_etf_fund_daily_em` | 1428x11 | 无需参数 | 基金代码, 基金简称, 类型, 2026-03-19-单位净值, 2026-03-19-累计净值, 2026-03-18-单位净值, 2026-03-18-累计净值, 增长值... (+3) |
| `fund_etf_hist_sina` | 0x0 | symbol="sz513180" |  |
| `fund_etf_spot_ths` | 1492x16 | 无需参数 | 序号, 基金代码, 基金名称, 当前-单位净值, 当前-累计净值, 前一日-单位净值, 前一日-累计净值, 增长值... (+7) |
| `fund_exchange_rank_em` | 1438x17 | 无需参数 | 序号, 基金代码, 基金简称, 类型, 日期, 单位净值, 累计净值, 近1周... (+7) |
| `fund_fee_em` | 0x0 | 无需参数 |  |
| `fund_financial_fund_daily_em` | 0x0 | 无需参数 |  |
| `fund_hk_fund_hist_em` | 1000x5 | 无需参数 | 净值日期, 单位净值, 日增长值, 日增长率, 单位 |
| `fund_hk_rank_em` | 154x18 | 无需参数 | 序号, 基金代码, 基金简称, 币种, 日期, 单位净值, 日增长率, 近1周... (+7) |
| `fund_hold_structure_em` | 44x7 | 无需参数 | 序号, 截止日期, 基金家数, 机构持有比列, 个人持有比列, 内部持有比列, 总份额 |
| `fund_individual_achievement_xq` | 32x5 | 无需参数 | 业绩类型, 周期, 本产品区间收益, 本产品最大回撒, 周期收益同类排名 |
| `fund_individual_analysis_xq` | 3x6 | 无需参数 | 周期, 较同类风险收益比, 较同类抗风险波动, 年化波动率, 年化夏普比率, 最大回撤 |
| `fund_individual_basic_info_xq` | 14x2 | 无需参数 | item, value |
| `fund_individual_detail_hold_xq` | 3x2 | 无需参数 | 资产类型, 仓位占比 |
| `fund_individual_detail_info_xq` | 8x3 | 无需参数 | 费用类型, 条件或名称, 费用 |
| `fund_individual_profit_probability_xq` | 4x3 | 无需参数 | 持有时长, 盈利概率, 平均收益 |
| `fund_info_index_em` | 1710x18 | 无需参数 | 基金代码, 基金名称, 单位净值, 日期, 日增长率, 近1周, 近1月, 近3月... (+7) |
| `fund_lcx_rank_em` | 0x0 | 无需参数 |  |
| `fund_linghuo_position_lg` | 304x3 | 无需参数 | date, close, position |
| `fund_money_fund_daily_em` | 538x13 | 无需参数 | 基金代码, 基金简称, 2026-03-19-万份收益, 2026-03-19-7日年化%, 2026-03-19-单位净值, 2026-03-18-万份收益, 2026-03-18-7日年化%, 2026-03-18-单位净值... (+5) |
| `fund_money_rank_em` | 538x18 | 无需参数 | 序号, 基金代码, 基金简称, 日期, 万份收益, 年化收益率7日, 年化收益率14日, 年化收益率28日... (+7) |
| `fund_open_fund_info_em` | 3513x3 | 无需参数 | 净值日期, 单位净值, 日增长率 |
| `fund_open_fund_rank_em` | 19302x18 | 无需参数 | 序号, 基金代码, 基金简称, 日期, 单位净值, 累计净值, 日增长率, 近1周... (+7) |
| `fund_overview_em` | 1x18 | 无需参数 | 基金全称, 基金简称, 基金代码, 基金类型, 发行日期, 成立日期/规模, 净资产规模, 份额规模... (+7) |
| `fund_portfolio_bond_hold_em` | 68x6 | 无需参数 | 序号, 债券代码, 债券名称, 占净值比例, 持仓市值, 季度 |
| `fund_portfolio_change_em` | 40x6 | 无需参数 | 序号, 股票代码, 股票名称, 本期累计买入金额, 占期初基金资产净值比例, 季度 |
| `fund_portfolio_hold_em` | 369x7 | 无需参数 | 序号, 股票代码, 股票名称, 占净值比例, 持股数, 持仓市值, 季度 |
| `fund_portfolio_industry_allocation_em` | 48x5 | 无需参数 | 序号, 行业类别, 占净值比例, 市值, 截止时间 |
| `fund_rating_all` | 15437x11 | 无需参数 | 代码, 简称, 基金经理, 基金公司, 5星评级家数, 上海证券, 招商证券, 济安金信... (+3) |
| `fund_rating_ja` | 7723x14 | 无需参数 | 代码, 简称, 基金经理, 基金公司, 3年期评级-3年评级, 3年期评级-较上期, 单位净值, 日期... (+6) |
| `fund_rating_sh` | 4817x16 | 无需参数 | 代码, 简称, 基金经理, 基金公司, 3年期评级-3年评级, 3年期评级-较上期, 5年期评级-5年评级, 5年期评级-较上期... (+7) |
| `fund_rating_zs` | 3324x13 | 无需参数 | 代码, 简称, 基金经理, 基金公司, 3年期评级-3年评级, 3年期评级-较上期, 单位净值, 日期... (+5) |
| `fund_report_asset_allocation_cninfo` | 74x6 | 无需参数 | 报告期, 基金覆盖家数, 股票权益类占净资产比例, 债券固定收益类占净资产比例, 现金货币类占净资产比例, 基金市场净资产规模 |
| `fund_report_industry_allocation_cninfo` | 19x6 | 无需参数 | 行业编码, 证监会行业名称, 报告期, 基金覆盖家数, 行业规模, 占净资产比例 |
| `fund_report_stock_cninfo` | 3956x7 | 无需参数 | 序号, 股票代码, 股票简称, 报告期, 基金覆盖家数, 持股总数, 持股总市值 |
| `fund_scale_change_em` | 111x7 | 无需参数 | 序号, 截止日期, 基金家数, 期间申购, 期间赎回, 期末总份额, 期末净资产 |
| `fund_scale_close_sina` | 179x9 | 无需参数 | 序号, 基金代码, 基金简称, 单位净值, 总募集规模, 最近总份额, 成立日期, 基金经理... (+1) |
| `fund_scale_structured_sina` | 402x9 | 无需参数 | 序号, 基金代码, 基金简称, 单位净值, 总募集规模, 最近总份额, 成立日期, 基金经理... (+1) |
| `fund_stock_position_lg` | 426x3 | 无需参数 | date, close, position |

## 期权数据

✅ 34 成功 | ❌ 5 失败 | ⏱ 7 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `option_cffex_hs300_daily_sina` | 54x6 | 无需参数 | date, open, high, low, close, volume |
| `option_cffex_hs300_list_sina` | dict(1) | 无需参数 |  |
| `option_cffex_hs300_spot_sina` | 33x17 | 无需参数 | 看涨合约-买量, 看涨合约-买价, 看涨合约-最新价, 看涨合约-卖价, 看涨合约-卖量, 看涨合约-持仓量, 看涨合约-涨跌, 行权价... (+7) |
| `option_cffex_sz50_daily_sina` | 45x6 | 无需参数 | date, open, high, low, close, volume |
| `option_cffex_sz50_list_sina` | dict(1) | 无需参数 |  |
| `option_cffex_sz50_spot_sina` | 21x17 | 无需参数 | 看涨合约-买量, 看涨合约-买价, 看涨合约-最新价, 看涨合约-卖价, 看涨合约-卖量, 看涨合约-持仓量, 看涨合约-涨跌, 行权价... (+7) |
| `option_cffex_zz1000_daily_sina` | 19x6 | 无需参数 | date, open, high, low, close, volume |
| `option_cffex_zz1000_list_sina` | dict(1) | 无需参数 |  |
| `option_cffex_zz1000_spot_sina` | 20x17 | 无需参数 | 看涨合约-买量, 看涨合约-买价, 看涨合约-最新价, 看涨合约-卖价, 看涨合约-卖量, 看涨合约-持仓量, 看涨合约-涨跌, 行权价... (+7) |
| `option_commodity_contract_sina` | 5x2 | 无需参数 | 序号, 合约 |
| `option_commodity_contract_table_sina` | 29x17 | 无需参数 | 看涨合约-买量, 看涨合约-买价, 看涨合约-最新价, 看涨合约-卖价, 看涨合约-卖量, 看涨合约-持仓量, 看涨合约-涨跌, 行权价... (+7) |
| `option_commodity_hist_sina` | 146x6 | 无需参数 | date, open, high, low, close, volume |
| `option_current_day_sse` | 716x11 | 无需参数 | 合约编码, 合约交易代码, 合约简称, 标的券名称及代码, 类型, 行权价, 合约单位, 期权行权日... (+3) |
| `option_daily_stats_sse` | 5x12 | 无需参数 | 合约标的代码, 合约标的名称, 合约数量, 总成交额, 总成交量, 认购成交量, 认沽成交量, 认沽/认购... (+4) |
| `option_daily_stats_szse` | 4x10 | 无需参数 | 合约标的代码, 合约标的名称, 成交量, 认购成交量, 认沽成交量, 认沽/认购持仓比, 未平仓合约总数, 未平仓认购合约数... (+2) |
| `option_finance_minute_sina` | 1222x5 | 无需参数 | date, time, price, average_price, volume |
| `option_finance_sse_underlying` | 1x9 | 无需参数 | 代码, 名称, 当前价, 涨跌, 涨跌幅, 振幅, 成交量(手), 成交额(万元)... (+1) |
| `option_hist_czce` | 138x16 | 无需参数 | 合约代码, 昨结算, 今开盘, 最高价, 最低价, 今收盘, 今结算, 涨跌1... (+7) |
| `option_hist_gfex` | 651x17 | 无需参数 | 商品名称, 合约名称, 开盘价, 最高价, 最低价, 收盘价, 前结算价, 结算价... (+7) |
| `option_hist_shfe` | 520x15 | 无需参数 | 合约代码, 开盘价, 最高价, 最低价, 收盘价, 前结算价, 结算价, 涨跌1... (+7) |
| `option_hist_yearly_czce` | 28570x17 | 无需参数 | 交易日期  , 合约代码   , 昨结算    , 今开盘    , 最高价    , 最低价    , 今收盘    , 今结算    ... (+7) |
| `option_lhb_em` | 7x10 | 无需参数 | 交易类型, 交易日期, 证券代码, 标的名称, 名次, 机构, 交易量, 增减... (+2) |
| `option_margin` | 420x12 | 无需参数 | 合约标的, 合约代码, 结算价, 交易乘数, 买方权利金, 卖方保证金, 手续费单位, 开仓手续费... (+4) |
| `option_margin_symbol` | 60x2 | 无需参数 | symbol, url |
| `option_risk_indicator_sse` | 396x10 | 无需参数 | TRADE_DATE, SECURITY_ID, CONTRACT_ID, CONTRACT_SYMBOL, DELTA_VALUE, THETA_VALUE, GAMMA_VALUE, VEGA_VALUE... (+2) |
| `option_sse_daily_sina` | 23x6 | 无需参数 | 日期, 开盘, 最高, 最低, 收盘, 成交量 |
| `option_sse_expire_day_sina` | tuple | 无需参数 |  |
| `option_sse_greeks_sina` | 1x2 | 无需参数 | 字段, 值 |
| `option_sse_list_sina` | list(4) | 无需参数 |  |
| `option_sse_minute_sina` | 0x6 | 无需参数 | 日期, 时间, 价格, 成交, 持仓, 均价 |
| `option_sse_spot_price_sina` | 1x2 | 无需参数 | 字段, 值 |
| `option_sse_underlying_spot_price_sina` | 33x2 | 无需参数 | 字段, 值 |
| `option_vol_gfex` | 7x2 | 无需参数 | 合约系列, 隐含波动率 |
| `option_vol_shfe` | 7x7 | 无需参数 | 合约系列, 成交量, 持仓量, 持仓量变化, 成交额, 行权量, 隐含波动率 |

## 期货数据

✅ 30 成功 | ❌ 18 失败 | ⏱ 10 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `futures_contract_detail` | 15x2 | 无需参数 | item, value |
| `futures_contract_info_cffex` | 836x12 | 无需参数 | 合约代码, 合约月份, 挂盘基准价, 上市日, 最后交易日, 涨停板幅度, 跌停板幅度, 涨停板价位... (+4) |
| `futures_contract_info_czce` | 218x40 | 无需参数 | 产品名称, 合约代码, 产品代码, 产品类型, 交易所MIC编码, 交易场所, 交易时间节假日除外, 交易国家ISO编码... (+7) |
| `futures_contract_info_gfex` | 46x7 | 无需参数 | 品种, 合约代码, 交易单位, 最小变动单位, 开始交易日, 最后交易日, 最后交割日 |
| `futures_contract_info_ine` | 62x7 | 无需参数 | 合约代码, 上市日, 到期日, 开始交割日, 最后交割日, 挂牌基准价, 交易日 |
| `futures_contract_info_shfe` | 276x8 | 无需参数 | 合约代码, 上市日, 到期日, 开始交割日, 最后交割日, 挂牌基准价, 交易日, 更新时间 |
| `futures_delivery_czce` | 7x3 | 无需参数 | 品种, 交割数量, 交割额 |
| `futures_delivery_match_czce` | 27x7 | 无需参数 | 卖方会员, 卖方会员-会员简称, 买方会员, 买方会员-会员简称, 交割量, 配对日期, 合约代码 |
| `futures_fees_info` | 867x38 | 无需参数 | 交易所, 合约代码, 合约名称, 品种代码, 品种名称, 合约乘数, 最小跳动, 开仓费率... (+7) |
| `futures_foreign_commodity_subscribe_exchange_symbol` | list(30) | 无需参数 |  |
| `futures_foreign_detail` | 4x6 | 无需参数 | 0, 1, 2, 3, 4, 5 |
| `futures_foreign_hist` | 2540x8 | 无需参数 | date, open, high, low, close, volume, position, s |
| `futures_gfex_warehouse_receipt` | dict(2) | 无需参数 |  |
| `futures_hog_core` | 367x2 | 无需参数 | date, value |
| `futures_hog_cost` | 367x2 | 无需参数 | date, value |
| `futures_hog_supply` | 90x2 | 无需参数 | date, value |
| `futures_hold_pos_sina` | 0x4 | 无需参数 | 名次, 会员简称, 成交量, 比上交易增减 |
| `futures_hq_subscribe_exchange_symbol` | 30x2 | 无需参数 | symbol, code |
| `futures_inventory_em` | 65x3 | 无需参数 | 日期, 库存, 增减 |
| `futures_main_sina` | 4078x8 | symbol="V0" | 日期, 开盘价, 最高价, 最低价, 收盘价, 成交量, 持仓量, 动态结算价 |
| `futures_rule` | 122x10 | 无需参数 | 交易所, 品种, 代码, 交易保证金比例, 涨跌停板幅度, 合约乘数, 最小变动价位, 限价单每笔最大下单手数... (+2) |
| `futures_shfe_warehouse_receipt` | dict(19) | 无需参数 |  |
| `futures_spot_price` | 0x0 | 无需参数 |  |
| `futures_spot_stock` | 5x10 | 无需参数 | 商品名称, 08-31, 09-30, 10-31, 11-30, 12-31, 最新价格, 近半年涨跌幅... (+2) |
| `futures_stock_shfe_js` | 0x0 | 无需参数 |  |
| `futures_symbol_mark` | 86x3 | 无需参数 | exchange, symbol, mark |
| `futures_to_spot_czce` | 2x2 | 无需参数 | 合约代码, 合约数量 |
| `futures_zh_daily_sina` | 4078x8 | symbol="V0" | date, open, high, low, close, volume, hold, settle |
| `futures_zh_minute_sina` | 1023x7 | symbol="V2501", period="5" | datetime, open, high, low, close, volume, hold |
| `futures_zh_realtime` | 13x23 | 无需参数 | symbol, exchange, name, trade, settlement, presettlement, open, high... (+7) |

## 债券数据

✅ 24 成功 | ❌ 7 失败 | ⏱ 8 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `bond_buy_back_hist_em` | 4802x7 | 无需参数 | 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额 |
| `bond_cash_summary_sse` | 10x5 | 无需参数 | 债券现货, 托管只数, 托管市值, 托管面值, 数据日期 |
| `bond_cb_adj_logs_jsl` | 3x6 | 无需参数 | 转债名称, 股东大会日, 下修前转股价, 下修后转股价, 新转股价生效日期, 下修底价 |
| `bond_cb_index_jsl` | 250x30 | 无需参数 | price_dt, price, amount, volume, count, increase_val, increase_rt, avg_price... (+7) |
| `bond_cb_jsl` | 30x23 | 无需参数 | 代码, 转债名称, 现价, 涨跌幅, 正股代码, 正股名称, 正股价, 正股涨跌... (+7) |
| `bond_cb_profile_sina` | 25x2 | 无需参数 | item, value |
| `bond_cb_redeem_jsl` | 362x18 | 无需参数 | 代码, 名称, 现价, 正股代码, 正股名称, 规模, 剩余规模, 转股起始日... (+7) |
| `bond_cb_summary_sina` | 15x2 | 无需参数 | item, value |
| `bond_china_close_return_map` | 75x3 | 无需参数 | value, cnLabel, enLabel |
| `bond_composite_index_cbond` | 6055x2 | 无需参数 | date, value |
| `bond_corporate_issue_cninfo` | 431x17 | 无需参数 | 债券代码, 债券简称, 公告日期, 交易所网上发行起始日, 交易所网上发行终止日, 计划发行总量, 实际发行总量, 发行面值... (+7) |
| `bond_cov_issue_cninfo` | 15x31 | 无需参数 | 债券代码, 债券简称, 公告日期, 发行起始日, 发行终止日, 计划发行总量, 实际发行总量, 发行面值... (+7) |
| `bond_cov_stock_issue_cninfo` | 103x10 | 无需参数 | 债券代码, 债券简称, 公告日期, 转股代码, 转股简称, 转股价格, 自愿转换期起始日, 自愿转换期终止日... (+2) |
| `bond_info_cm_query` | 27x2 | 无需参数 | name, code |
| `bond_local_government_issue_cninfo` | 1449x15 | 无需参数 | 债券代码, 债券简称, 发行起始日, 发行终止日, 计划发行总量, 实际发行总量, 发行价格, 单位面值... (+7) |
| `bond_spot_deal` | 3061x6 | 无需参数 | 债券简称, 成交净价, 最新收益率, 涨跌, 加权收益率, 交易量 |
| `bond_spot_quote` | 15x6 | 无需参数 | 报价机构, 债券简称, 买入净价, 卖出净价, 买入收益率, 卖出收益率 |
| `bond_treasure_issue_cninfo` | 104x15 | 无需参数 | 债券代码, 债券简称, 发行起始日, 发行终止日, 计划发行总量, 实际发行总量, 发行价格, 单位面值... (+7) |
| `bond_zh_cov_info` | 1x72 | 无需参数 | SECURITY_CODE, SECUCODE, TRADE_MARKET, SECURITY_NAME_ABBR, DELIST_DATE, LISTING_DATE, CONVERT_STOCK_CODE, BOND_EXPIRE... (+7) |
| `bond_zh_cov_info_ths` | 915x16 | 无需参数 | 债券代码, 债券简称, 申购日期, 申购代码, 原股东配售码, 每股获配额, 计划发行量, 实际发行量... (+7) |
| `bond_zh_cov_value_analysis` | 1457x6 | 无需参数 | 日期, 收盘价, 纯债价值, 转股价值, 纯债溢价率, 转股溢价率 |
| `bond_zh_hs_cov_daily` | 4806x6 | 无需参数 | date, open, high, low, close, volume |
| `bond_zh_hs_cov_spot` | 378x15 | 无需参数 | symbol, name, trade, pricechange, changepercent, buy, sell, settlement... (+7) |
| `bond_zh_hs_daily` | 4806x6 | 无需参数 | date, open, high, low, close, volume |

## 其他

✅ 24 成功 | ❌ 18 失败 | ⏱ 5 超时 | ⊘ 2 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `business_value_artist` | 100x8 | 无需参数 | 排名, 艺人, 商业价值, 专业热度, 关注热度, 预测热度, 美誉度, 统计日期 |
| `drewry_wci_index` | 517x2 | 无需参数 | date, wci |
| `get_cffex_daily` | 4x12 | 无需参数 | symbol, date, open, high, low, close, volume, open_interest... (+4) |
| `get_cffex_rank_table` | dict(12) | 无需参数 |  |
| `get_futures_daily` | 21x12 | 无需参数 | symbol, date, open, high, low, close, volume, open_interest... (+4) |
| `get_gfex_daily` | 5x12 | 无需参数 | symbol, date, open, high, low, close, volume, open_interest... (+4) |
| `get_ine_daily` | 65x12 | 无需参数 | symbol, date, open, high, low, close, volume, open_interest... (+4) |
| `get_qhkc_fund_money_change` | 61x4 | 无需参数 | name, value, ratio, date |
| `get_qhkc_index_trend` | 893x5 | 无需参数 | broker, grade, money, open_order, variety |
| `get_rank_table_czce` | dict(117) | 无需参数 |  |
| `get_receipt` | 0x0 | 无需参数 |  |
| `get_shfe_daily` | 248x13 | 无需参数 | index, symbol, date, open, high, low, close, volume... (+5) |
| `get_shfe_rank_table` | dict(72) | 无需参数 |  |
| `hf_sp_500` | 222026x6 | 无需参数 | date, open, high, low, close, price |
| `match_main_contract` | str | 无需参数 |  |
| `nlp_answer` | str | 无需参数 |  |
| `nlp_ownthink` | str | 无需参数 |  |
| `qdii_a_index_jsl` | 20x17 | 无需参数 | 代码, 名称, 现价, 涨幅, 成交, 场内份额, 场内新增, 净值... (+7) |
| `qdii_e_index_jsl` | 20x18 | 无需参数 | 代码, 名称, 现价, 涨幅, 成交, 场内份额, 场内新增, T-2净值... (+7) |
| `repo_rate_hist` | 17x7 | 无需参数 | date, FR001, FR007, FR014, FDR001, FDR007, FDR014 |
| `repo_rate_query` | 749x4 | 无需参数 | date, FR001, FR007, FR014 |
| `rv_from_futures_zh_minute_sina` | 1023x6 | 无需参数 | Open, High, Low, Close, volume, hold |
| `video_tv` | 10x9 | 无需参数 | 排序, 名称, 类型, 播映指数, 媒体热度, 用户热度, 好评度, 观看度... (+1) |
| `video_variety_show` | 10x9 | 无需参数 | 排序, 名称, 类型, 播映指数, 媒体热度, 用户热度, 好评度, 观看度... (+1) |

## 另类数据

✅ 23 成功 | ❌ 2 失败 | ⏱ 3 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `air_city_table` | 168x7 | 无需参数 | 序号, 省份, 城市, AQI, 空气质量, PM2.5浓度, 首要污染物 |
| `air_quality_hist` | 10x15 | city="北京", period="day", start_date="20250101", end_date="20250110" | aqi, pm2_5, pm10, co, no2, o3, so2, complexindex... (+7) |
| `air_quality_rank` | 168x7 | 无需参数 | 降序, 省份, 城市, AQI, 空气质量, PM2.5浓度, 首要污染物 |
| `air_quality_watch_point` | 31x8 | city="北京" | pointname, aqi, pm2_5, pm10, no2, so2, o3, co |
| `car_market_country_cpca` | 12x8 | 无需参数 | 月份, 其他欧系, 德系, 日系, 法系, 美系, 自主, 韩系 |
| `car_market_fuel_cpca` | 12x3 | 无需参数 | 月份, 2025年, 2026年 |
| `car_market_man_rank_cpca` | 10x3 | 无需参数 | 厂商, 2025年2月, 2026年2月 |
| `car_market_segment_cpca` | 12x6 | 无需参数 | 月份, A00, A0, A, B, C |
| `car_market_total_cpca` | 12x3 | 无需参数 | 月份, 2025年, 2026年 |
| `car_sale_rank_gasgoo` | 50x7 | 无需参数 | 厂商, 2021-9, 9月同比, 9月环比, 2021-1到9, 2020-1到9, 2019-1到9 |
| `forbes_rank` | 99x6 | 无需参数 | 1, 沈南鹏, 男, 53, 红杉中国, 创始及执行合伙人 |
| `migration_scale_baidu` | 939x2 | 无需参数 | 日期, 迁徙规模指数 |
| `movie_boxoffice_cinema_daily` | 100x7 | 无需参数 | 排序, 影院名称, 单日票房, 单日场次, 场均人次, 场均票价, 上座率 |
| `movie_boxoffice_cinema_weekly` | 100x7 | 无需参数 | 排序, 影院名称, 当周票房, 单银幕票房, 场均人次, 单日单厅票房, 单日单厅场次 |
| `movie_boxoffice_daily` | 10x9 | 无需参数 | 排序, 影片名称, 单日票房, 环比变化, 累计票房, 平均票价, 场均人次, 口碑指数... (+1) |
| `movie_boxoffice_monthly` | 11x9 | 无需参数 | 排序, 影片名称, 单月票房, 月度占比, 平均票价, 场均人次, 上映日期, 口碑指数... (+1) |
| `movie_boxoffice_realtime` | 11x6 | 无需参数 | 排序, 影片名称, 实时票房, 票房占比, 上映天数, 累计票房 |
| `movie_boxoffice_weekly` | 10x10 | 无需参数 | 排序, 影片名称, 排名变化, 单周票房, 环比变化, 累计票房, 平均票价, 场均人次... (+2) |
| `movie_boxoffice_yearly` | 25x8 | 无需参数 | 排序, 影片名称, 类型, 总票房, 平均票价, 场均人次, 国家及地区, 上映日期 |
| `movie_boxoffice_yearly_first_week` | 8x9 | 无需参数 | 排序, 影片名称, 类型, 首周票房, 占总票房比重, 场均人次, 国家及地区, 上映日期... (+1) |
| `online_value_artist` | 100x8 | 无需参数 | 排名, 艺人, 流量价值, 专业热度, 关注热度, 预测热度, 带货力, 统计日期 |
| `sunrise_daily` | 1x14 | 无需参数 | date, Apr, Sunrise, Sunset, Length, Diff., Start, End... (+6) |
| `sunrise_monthly` | 30x14 | 无需参数 | date, Apr, Sunrise, Sunset, Length, Diff., Start, End... (+6) |

## 现货数据

✅ 7 成功 | ❌ 0 失败 | ⏱ 9 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `spot_golden_benchmark_sge` | 2409x3 | 无需参数 | 交易时间, 晚盘价, 早盘价 |
| `spot_goods` | 3983x4 | 无需参数 | 日期, 指数, 涨跌额, 涨跌幅 |
| `spot_hist_sge` | 2243x5 | 无需参数 | date, open, close, low, high |
| `spot_quotations_sge` | 542x4 | 无需参数 | 品种, 时间, 现价, 更新时间 |
| `spot_silver_benchmark_sge` | 1561x3 | 无需参数 | 交易时间, 晚盘价, 早盘价 |
| `spot_soybean_price_soozhu` | 15x2 | 无需参数 | 日期, 价格 |
| `spot_symbol_table_sge` | 17x2 | 无需参数 | 序号, 品种 |

## 货币数据

✅ 4 成功 | ❌ 3 失败 | ⏱ 1 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `currency_boc_sina` | 180x6 | 无需参数 | 日期, 中行汇买价, 中行钞买价, 中行钞卖价/汇卖价, 央行中间价, 中行折算价 |
| `currency_currencies` | 0x0 | 无需参数 |  |
| `currency_pair_map` | 193x2 | 无需参数 | name, code |
| `currency_time_series` | 0x1 | 无需参数 | date |

## 外汇数据

✅ 4 成功 | ❌ 1 失败 | ⏱ 0 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `fx_pair_quote` | 16x3 | 无需参数 | 货币对, 买报价, 卖报价 |
| `fx_quote_baidu` | 166x5 | 无需参数 | 代码, 名称, 最新价, 涨跌额, 涨跌幅 |
| `fx_spot_quote` | 25x3 | 无需参数 | 货币对, 买报价, 卖报价 |
| `fx_swap_quote` | 25x7 | 无需参数 | 货币对, 1周, 1月, 3月, 6月, 9月, 1年 |

## 新闻资讯

✅ 4 成功 | ❌ 0 失败 | ⏱ 1 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `news_economic_baidu` | 99x8 | 无需参数 | 日期, 时间, 地区, 事件, 公布, 预期, 前值, 重要性 |
| `news_report_time_baidu` | 67x7 | 无需参数 | 股票代码, 股票简称, 交易所, 财报类型, 发布时间, 市值, 发布日期 |
| `news_trade_notify_dividend_baidu` | 15x9 | 无需参数 | 股票代码, 除权日, 分红, 送股, 转增, 实物, 交易所, 股票简称... (+1) |
| `news_trade_notify_suspend_baidu` | 8x12 | 无需参数 | 股票代码, 股票简称, 交易所代码, 停牌时间, 复牌时间, 停牌事项说明, 市值, 公告日期... (+4) |

## 申万指数

✅ 4 成功 | ❌ 0 失败 | ⏱ 0 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `sw_index_first_info` | 31x7 | 无需参数 | 行业代码, 行业名称, 成份个数, 静态市盈率, TTM(滚动)市盈率, 市净率, 静态股息率 |
| `sw_index_second_info` | 124x8 | 无需参数 | 行业代码, 行业名称, 上级行业, 成份个数, 静态市盈率, TTM(滚动)市盈率, 市净率, 静态股息率 |
| `sw_index_third_cons` | 123x17 | 无需参数 | 序号, 股票代码, 股票简称, 纳入时间, 申万1级, 申万2级, 申万3级, 价格... (+7) |
| `sw_index_third_info` | 258x8 | 无需参数 | 行业代码, 行业名称, 上级行业, 成份个数, 静态市盈率, TTM(滚动)市盈率, 市净率, 静态股息率 |

## 波动率/学术

✅ 3 成功 | ❌ 2 失败 | ⏱ 0 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `article_epu_index` | 347x3 | 无需参数 | year, month, China_Policy_Index |
| `article_ff_crr` | 32x4 | 无需参数 | item, January  2026, Last 3  Months, Last 12  Months |
| `article_rlab_rv` | Series(7089) | 无需参数 |  |

## 加密货币

✅ 3 成功 | ❌ 0 失败 | ⏱ 0 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `crypto_bitcoin_cme` | 5x8 | 无需参数 | 商品, 类型, 电子交易合约, 场内成交合约, 场外成交合约, 成交量, 未平仓合约, 持仓变化 |
| `crypto_bitcoin_hold_report` | 59x14 | 无需参数 | 代码, 公司名称-英文, 公司名称-中文, 国家/地区, 市值, 比特币占市值比重, 持仓成本, 持仓占比... (+6) |
| `crypto_js_spot` | 10x9 | 无需参数 | 市场, 交易品种, 最近报价, 涨跌额, 涨跌幅, 24小时最高, 24小时最低, 24小时成交量... (+1) |

## 工具箱

✅ 1 成功 | ❌ 0 失败 | ⏱ 0 超时 | ⊘ 0 跳过

| 接口名称 | 数据规模 | 示例参数 | 返回字段 |
|---------|---------|---------|--------|
| `tool_trade_date_hist_sina` | 8797x1 | 无需参数 | trade_date |

## 分类统计汇总

| 分类 | 成功 | 失败 | 超时 | 跳过 | 成功率 |
|-----|------|------|------|------|-------|
| 债券数据 | 24 | 7 | 8 | 0 | 62% |
| 其他 | 24 | 18 | 5 | 2 | 49% |
| 利率数据 | 0 | 0 | 1 | 0 | 0% |
| 加密货币 | 3 | 0 | 0 | 0 | 100% |
| 另类数据 | 23 | 2 | 3 | 0 | 82% |
| 基金数据 | 48 | 3 | 17 | 0 | 71% |
| 外汇数据 | 4 | 1 | 0 | 0 | 80% |
| 奇货可查 | 0 | 0 | 0 | 2 | 0% |
| 宏观数据 | 108 | 3 | 113 | 2 | 48% |
| 工具箱 | 1 | 0 | 0 | 0 | 100% |
| 指数数据 | 62 | 5 | 12 | 0 | 78% |
| 新闻资讯 | 4 | 0 | 1 | 0 | 80% |
| 期权数据 | 34 | 5 | 7 | 0 | 74% |
| 期货数据 | 30 | 18 | 10 | 0 | 52% |
| 波动率/学术 | 3 | 2 | 0 | 0 | 60% |
| 现货数据 | 7 | 0 | 9 | 0 | 44% |
| 申万指数 | 4 | 0 | 0 | 0 | 100% |
| 私募基金 | 0 | 0 | 0 | 14 | 0% |
| 股票数据 | 211 | 33 | 158 | 0 | 52% |
| 能源数据 | 0 | 2 | 6 | 0 | 0% |
| 货币数据 | 4 | 3 | 1 | 0 | 50% |
| 银行数据 | 0 | 0 | 1 | 0 | 0% |

## 失败接口列表

### 超时接口

- `air_quality_hebei`
- `bank_fjcf_table_detail`
- `bond_china_yield`
- `bond_cov_comparison`
- `bond_deal_summary_sse`
- `bond_debt_nafmii`
- `bond_info_cm`
- `bond_new_composite_index_cbond`
- `bond_zh_cov`
- `bond_zh_us_rate`
- `currency_boc_safe`
- `energy_carbon_domestic`
- `energy_carbon_eu`
- `energy_carbon_hb`
- `energy_carbon_sz`
- `energy_oil_detail`
- `energy_oil_hist`
- `forex_spot_em`
- `fund_etf_fund_info_em`
- `fund_etf_spot_em`
- `fund_fh_em`
- `fund_fh_rank_em`
- `fund_graded_fund_daily_em`
- `fund_graded_fund_info_em`
- `fund_lof_hist_em`
- `fund_lof_hist_min_em`
- `fund_lof_spot_em`
- `fund_manager_em`
- `fund_money_fund_info_em`
- `fund_name_em`
- `fund_new_found_em`
- `fund_open_fund_daily_em`
- `fund_purchase_em`
- `fund_scale_open_sina`
- `fund_value_estimation_em`
- `futures_comex_inventory`
- `futures_comm_info`
- `futures_display_main_sina`
- `futures_gfex_position_rank`
- `futures_global_spot_em`
- `futures_hist_em`
- `futures_hist_table_em`
- `futures_news_shmet`
- `futures_spot_price_daily`
- `futures_warehouse_receipt_czce`
- `get_dce_rank_table`
- `get_qhkc_index`
- `get_qhkc_index_profit_loss`
- `get_us_stock_name`
- `hurun_rank`
- `index_code_id_map_em`
- `index_detail_hist_adjust_cni`
- `index_eri`
- `index_kq_fz`
- `index_news_sentiment_scope`
- `index_option_50etf_qvix`
- `index_realtime_sw`
- `index_stock_cons_weight_csindex`
- `index_stock_info`
- `index_ti_cx`
- `index_zh_a_hist`
- `index_zh_a_hist_min_em`
- `macro_bank_australia_interest_rate`
- `macro_bank_english_interest_rate`
- `macro_bank_euro_interest_rate`
- `macro_bank_india_interest_rate`
- `macro_bank_japan_interest_rate`
- `macro_bank_russia_interest_rate`
- `macro_bank_usa_interest_rate`
- `macro_canada_core_cpi_monthly`
- `macro_canada_cpi_monthly`
- `macro_canada_gdp_monthly`
- `macro_canada_trade`
- `macro_china_agricultural_index`
- `macro_china_agricultural_product`
- `macro_china_au_report`
- `macro_china_bdti_index`
- `macro_china_bond_public`
- `macro_china_bsi_index`
- `macro_china_central_bank_balance`
- `macro_china_commodity_price_index`
- `macro_china_construction_index`
- `macro_china_construction_price_index`
- `macro_china_cpi_monthly`
- `macro_china_cpi_yearly`
- `macro_china_energy_index`
- `macro_china_enterprise_boom_index`
- `macro_china_exports_yoy`
- `macro_china_foreign_exchange_gold`
- `macro_china_fx_gold`
- `macro_china_hk_building_amount`
- `macro_china_hk_cpi`
- `macro_china_imports_yoy`
- `macro_china_industrial_production_yoy`
- `macro_china_insurance`
- `macro_china_lpi_index`
- `macro_china_lpr`
- `macro_china_m2_yearly`
- `macro_china_money_supply`
- `macro_china_non_man_pmi`
- `macro_china_pmi_yearly`
- `macro_china_postal_telecommunicational`
- `macro_china_ppi_yearly`
- `macro_china_real_estate`
- `macro_china_retail_price_index`
- `macro_china_shibor_all`
- `macro_china_shrzgm`
- `macro_china_society_traffic_volume`
- `macro_china_trade_balance`
- `macro_china_urban_unemployment`
- `macro_china_vegetable_basket`
- `macro_china_yw_electronic_index`
- `macro_cons_gold`
- `macro_cons_opec_month`
- `macro_cons_silver`
- `macro_euro_cpi_mom`
- `macro_euro_cpi_yoy`
- `macro_euro_gdp_yoy`
- `macro_euro_industrial_production_mom`
- `macro_euro_manufacturing_pmi`
- `macro_euro_ppi_mom`
- `macro_euro_retail_sales_mom`
- `macro_euro_sentix_investor_confidence`
- `macro_euro_services_pmi`
- `macro_euro_trade_balance`
- `macro_euro_unemployment_rate_mom`
- `macro_germany_cpi_monthly`
- `macro_global_sox_index`
- `macro_info_ws`
- `macro_japan_head_indicator`
- `macro_shipping_bci`
- `macro_shipping_bcti`
- `macro_shipping_bdi`
- `macro_shipping_bpi`
- `macro_uk_cpi_monthly`
- `macro_uk_cpi_yearly`
- `macro_uk_retail_yearly`
- `macro_uk_trade`
- `macro_usa_adp_employment`
- `macro_usa_api_crude_stock`
- `macro_usa_building_permits`
- `macro_usa_business_inventories`
- `macro_usa_cb_consumer_confidence`
- `macro_usa_cme_merchant_goods_holding`
- `macro_usa_core_cpi_monthly`
- `macro_usa_core_pce_price`
- `macro_usa_core_ppi`
- `macro_usa_cpi_monthly`
- `macro_usa_durable_goods_orders`
- `macro_usa_eia_crude_rate`
- `macro_usa_exist_home_sales`
- `macro_usa_export_price`
- `macro_usa_factory_orders`
- `macro_usa_house_price_index`
- `macro_usa_house_starts`
- `macro_usa_import_price`
- `macro_usa_industrial_production`
- `macro_usa_initial_jobless`
- `macro_usa_ism_non_pmi`
- `macro_usa_ism_pmi`
- `macro_usa_job_cuts`
- `macro_usa_michigan_consumer_sentiment`
- `macro_usa_nahb_house_market_index`
- `macro_usa_new_home_sales`
- `macro_usa_nfib_small_business`
- `macro_usa_non_farm`
- `macro_usa_pending_home_sales`
- `macro_usa_personal_spending`
- `macro_usa_pmi`
- `macro_usa_ppi`
- `macro_usa_retail_sales`
- `macro_usa_services_pmi`
- `macro_usa_spcs20`
- `macro_usa_trade_balance`
- `macro_usa_unemployment_rate`
- `news_cctv`
- `option_contract_info_ctp`
- `option_current_em`
- `option_finance_board`
- `option_minute_em`
- `option_premium_analysis_em`
- `option_risk_analysis_em`
- `option_value_analysis_em`
- `rate_interbank`
- `spot_corn_price_soozhu`
- `spot_hog_crossbred_soozhu`
- `spot_hog_lean_price_soozhu`
- `spot_hog_soozhu`
- `spot_hog_three_way_soozhu`
- `spot_hog_year_trend_soozhu`
- `spot_mixed_feed_soozhu`
- `spot_price_qh`
- `spot_price_table_qh`
- `stock_analyst_rank_em`
- `stock_balance_sheet_by_report_em`
- `stock_balance_sheet_by_yearly_em`
- `stock_bid_ask_em`
- `stock_bj_a_spot_em`
- `stock_board_concept_cons_em`
- `stock_board_concept_hist_em`
- `stock_board_concept_hist_min_em`
- `stock_board_concept_index_ths`
- `stock_board_concept_info_ths`
- `stock_board_concept_name_em`
- `stock_board_concept_name_ths`
- `stock_board_concept_spot_em`
- `stock_board_concept_summary_ths`
- `stock_board_industry_cons_em`
- `stock_board_industry_hist_em`
- `stock_board_industry_hist_min_em`
- `stock_board_industry_name_em`
- `stock_board_industry_spot_em`
- `stock_cash_flow_sheet_by_quarterly_em`
- `stock_cash_flow_sheet_by_report_em`
- `stock_cash_flow_sheet_by_yearly_em`
- `stock_classify_sina`
- `stock_comment_em`
- `stock_concept_fund_flow_hist`
- `stock_cy_a_spot_em`
- `stock_dxsyl_em`
- `stock_dzjy_hygtj`
- `stock_dzjy_hyyybtj`
- `stock_dzjy_sctj`
- `stock_dzjy_yybph`
- `stock_esg_hz_sina`
- `stock_esg_msci_sina`
- `stock_esg_rate_sina`
- `stock_esg_zd_sina`
- `stock_fhps_em`
- `stock_fund_flow_big_deal`
- `stock_fund_flow_individual`
- `stock_gddh_em`
- `stock_gdfx_free_holding_analyse_em`
- `stock_gdfx_free_holding_change_em`
- `stock_gdfx_free_holding_detail_em`
- `stock_gdfx_free_holding_statistics_em`
- `stock_gdfx_free_holding_teamwork_em`
- `stock_gdfx_holding_analyse_em`
- `stock_gdfx_holding_change_em`
- `stock_gdfx_holding_detail_em`
- `stock_gdfx_holding_statistics_em`
- `stock_gdfx_holding_teamwork_em`
- `stock_ggcg_em`
- `stock_gpzy_pledge_ratio_detail_em`
- `stock_gpzy_pledge_ratio_em`
- `stock_gpzy_profile_em`
- `stock_hk_ggt_components_em`
- `stock_hk_hot_rank_detail_realtime_em`
- `stock_hk_hot_rank_em`
- `stock_hk_index_daily_em`
- `stock_hk_indicator_eniu`
- `stock_hk_main_board_spot_em`
- `stock_hk_spot`
- `stock_hk_spot_em`
- `stock_hold_management_detail_em`
- `stock_hold_management_person_em`
- `stock_hot_deal_xq`
- `stock_hot_follow_xq`
- `stock_hot_tweet_xq`
- `stock_hsgt_board_rank_em`
- `stock_hsgt_hist_em`
- `stock_hsgt_hold_stock_em`
- `stock_hsgt_individual_em`
- `stock_hsgt_institution_statistics_em`
- `stock_hsgt_sh_hk_spot_em`
- `stock_index_pb_lg`
- `stock_individual_fund_flow_rank`
- `stock_info_a_code_name`
- `stock_info_bj_name_code`
- `stock_info_sh_name_code`
- `stock_ipo_declare_em`
- `stock_ipo_review_em`
- `stock_ipo_tutor_em`
- `stock_jgdy_detail_em`
- `stock_jgdy_tj_em`
- `stock_kc_a_spot_em`
- `stock_lh_yyb_capital`
- `stock_lh_yyb_most`
- `stock_lhb_detail_em`
- `stock_lhb_hyyyb_em`
- `stock_lhb_jgstatistic_em`
- `stock_lhb_stock_statistic_em`
- `stock_lhb_traderstatistic_em`
- `stock_lhb_yyb_detail_em`
- `stock_lhb_yybph_em`
- `stock_lhb_yytj_sina`
- `stock_lrb_em`
- `stock_main_fund_flow`
- `stock_margin_account_info`
- `stock_new_a_spot_em`
- `stock_news_em`
- `stock_news_main_cx`
- `stock_notice_report`
- `stock_pg_em`
- `stock_profit_forecast_em`
- `stock_profit_sheet_by_quarterly_em`
- `stock_profit_sheet_by_report_em`
- `stock_profit_sheet_by_yearly_em`
- `stock_qbzf_em`
- `stock_rank_cxd_ths`
- `stock_rank_ljqd_ths`
- `stock_rank_lxxd_ths`
- `stock_rank_xstp_ths`
- `stock_rank_xxtp_ths`
- `stock_register_all_em`
- `stock_register_bj`
- `stock_register_cyb`
- `stock_register_db`
- `stock_register_kcb`
- `stock_register_sh`
- `stock_register_sz`
- `stock_report_fund_hold`
- `stock_repurchase_em`
- `stock_restricted_release_detail_em`
- `stock_sector_fund_flow_summary`
- `stock_sh_a_spot_em`
- `stock_share_hold_change_szse`
- `stock_sns_sseinfo`
- `stock_sy_em`
- `stock_sy_yq_em`
- `stock_sz_a_spot_em`
- `stock_tfp_em`
- `stock_us_spot`
- `stock_us_spot_em`
- `stock_value_em`
- `stock_xgsglb_em`
- `stock_xgsr_ths`
- `stock_xjll_em`
- `stock_yjbb_em`
- `stock_yysj_em`
- `stock_yzxdr_em`
- `stock_zcfz_bj_em`
- `stock_zcfz_em`
- `stock_zdhtmx_em`
- `stock_zh_a_disclosure_relation_cninfo`
- `stock_zh_a_gdhs`
- `stock_zh_a_hist_tx`
- `stock_zh_a_new_em`
- `stock_zh_a_spot`
- `stock_zh_a_spot_em`
- `stock_zh_a_st_em`
- `stock_zh_a_stop_em`
- `stock_zh_a_tick_tx_js`
- `stock_zh_ab_comparison_em`
- `stock_zh_ah_daily`
- `stock_zh_ah_name`
- `stock_zh_ah_spot`
- `stock_zh_ah_spot_em`
- `stock_zh_b_spot_em`
- `stock_zh_index_spot_em`
- `stock_zh_kcb_report_em`
- `xincaifu_rank`

### 错误接口

| 接口 | 错误类型 | 错误信息 |
|-----|---------|--------|
| `article_oman_rv` | ConnectionError | HTTPSConnectionPool(host='realized.oxford-man.ox.ac.uk', port=443): Max retries  |
| `article_oman_rv_short` | ConnectionError | HTTPSConnectionPool(host='realized.oxford-man.ox.ac.uk', port=443): Max retries  |
| `bond_china_close_return` | KeyError | 'newDateValue' |
| `bond_info_detail_cm` | JSONDecodeError | Expecting value: line 1 column 9 (char 8) |
| `bond_sh_buy_back_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `bond_sz_buy_back_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `bond_zh_hs_cov_min` | TypeError | 'NoneType' object is not subscriptable |
| `bond_zh_hs_cov_pre_min` | TypeError | 'NoneType' object is not subscriptable |
| `bond_zh_hs_spot` | JSONDecodeError | No value to decode |
| `car_market_cate_cpca` | KeyError | '2026年' |
| `currency_convert` | KeyError | 'timestamp' |
| `currency_history` | KeyError | 'date' |
| `currency_latest` | KeyError | 'date' |
| `energy_carbon_bj` | AttributeError | 'NoneType' object has no attribute 'find' |
| `energy_carbon_gz` | ValueError | No tables found |
| `forex_hist_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `fred_md` | HTTPError | HTTP Error 403: Forbidden |
| `fred_qd` | HTTPError | HTTP Error 403: Forbidden |
| `fund_etf_hist_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `fund_etf_hist_min_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `fund_financial_fund_info_em` | TypeError | 'NoneType' object is not subscriptable |
| `futures_contract_detail_em` | AttributeError | 'NoneType' object has no attribute 'find' |
| `futures_contract_info_dce` | JSONDecodeError | Expecting value: line 1 column 1 (char 0) |
| `futures_dce_position_rank` | BadZipFile | File is not a zip file |
| `futures_dce_position_rank_other` | IndexError | list index out of range |
| `futures_delivery_dce` | ValueError | No tables found |
| `futures_delivery_match_dce` | ValueError | No tables found |
| `futures_delivery_shfe` | ConnectionError | HTTPSConnectionPool(host='tsite.shfe.com.cn', port=443): Max retries exceeded wi |
| `futures_foreign_commodity_realtime` | ValueError | Length mismatch: Expected axis has 1 elements, new values have 15 elements |
| `futures_global_hist_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `futures_index_ccidx` | TypeError | the JSON object must be str, bytes or bytearray, not dict |
| `futures_inventory_99` | JSONDecodeError | Expecting value: line 1 column 1 (char 0) |
| `futures_settlement_price_sgx` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `futures_spot_price_previous` | ValueError | No tables found |
| `futures_spot_sys` | AttributeError | 'NoneType' object has no attribute 'find_all' |
| `futures_to_spot_dce` | ValueError | No tables found |
| `futures_to_spot_shfe` | ConnectionError | HTTPSConnectionPool(host='tsite.shfe.com.cn', port=443): Max retries exceeded wi |
| `futures_warehouse_receipt_dce` | JSONDecodeError | Expecting value: line 1 column 1 (char 0) |
| `futures_zh_spot` | ValueError | Length mismatch: Expected axis has 1 elements, new values have 44 elements |
| `fx_c_swap_cm` | AttributeError | module 'ssl' has no attribute 'OP_LEGACY_SERVER_CONNECT' |
| `get_czce_daily` | BadZipFile | File is not a zip file |
| `get_dce_daily` | JSONDecodeError | Expecting value: line 1 column 1 (char 0) |
| `get_qhkc_fund_bs` | KeyError | 'datas1' |
| `get_qhkc_fund_position` | TypeError | list indices must be integers or slices, not str |
| `get_rank_sum` | BadZipFile | File is not a zip file |
| `get_rank_sum_daily` | BadZipFile | File is not a zip file |
| `get_roll_yield` | JSONDecodeError | Expecting value: line 1 column 1 (char 0) |
| `get_roll_yield_bar` | JSONDecodeError | Expecting value: line 1 column 1 (char 0) |
| `index_bloomberg_billionaires` | AttributeError | 'NoneType' object has no attribute 'find_all' |
| `index_bloomberg_billionaires_hist` | IndexError | list index out of range |
| `index_global_hist_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `index_global_hist_sina` | KeyError | 'OMX' |
| `index_global_spot_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `macro_china_supply_of_money` | JSONDecodeError | No value to decode |
| `macro_china_swap_rate` | ValueError | Length mismatch: Expected axis has 0 elements, new values have 17 elements |
| `macro_fx_sentiment` | KeyError | 'data' |
| `migration_area_baidu` | KeyError | 'value' |
| `option_comm_info` | AttributeError | 'NoneType' object has no attribute 'find_all' |
| `option_comm_symbol` | AttributeError | 'NoneType' object has no attribute 'find_all' |
| `option_current_day_szse` | ConnectionError | ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was  |
| `option_hist_dce` | JSONDecodeError | Expecting value: line 1 column 1 (char 0) |
| `option_sse_codes_sina` | ValueError | Length mismatch: Expected axis has 1 elements, new values have 2 elements |
| `pro_api` | Exception | api init error. |
| `qdii_e_comm_jsl` | KeyError | "None of [Index(['代码', '名称', '现价', '涨幅', '成交', '场内份额', '场内新增', 'T-2净值', '净值日期',  |
| `reits_hist_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `reits_hist_min_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `reits_realtime_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `rv_from_stock_zh_a_hist_min_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `stock_a_below_net_asset_statistics` | KeyError | 'marketId' |
| `stock_cg_lawsuit_cninfo` | KeyError | 'records' |
| `stock_concept_cons_futu` | JSONDecodeError | Expecting value: line 1 column 1 (char 0) |
| `stock_cyq_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `stock_gpzy_distribute_statistics_bank_em` | TypeError | 'NoneType' object is not subscriptable |
| `stock_gpzy_distribute_statistics_company_em` | TypeError | 'NoneType' object is not subscriptable |
| `stock_hk_gxl_lg` | KeyError | "['股息率'] not in index" |
| `stock_hsgt_individual_detail_em` | TypeError | 'NoneType' object is not subscriptable |
| `stock_individual_basic_info_hk_xq` | KeyError | 'data' |
| `stock_individual_basic_info_us_xq` | KeyError | 'data' |
| `stock_individual_basic_info_xq` | KeyError | 'data' |
| `stock_individual_fund_flow_rank_async` | ServerDisconnectedError | Server disconnected |
| `stock_individual_spot_xq` | KeyError | 'data' |
| `stock_industry_pe_ratio_cninfo` | ValueError | Length mismatch: Expected axis has 0 elements, new values have 12 elements |
| `stock_intraday_sina` | KeyError | 'ticktime' |
| `stock_ipo_benefit_ths` | AttributeError | 'NoneType' object has no attribute 'text' |
| `stock_margin_szse` | TypeError | stock_margin_szse() got an unexpected keyword argument 'start_date' |
| `stock_new_gh_cninfo` | ValueError | Length mismatch: Expected axis has 0 elements, new values have 6 elements |
| `stock_price_js` | JSONDecodeError | Expecting value: line 1 column 1 (char 0) |
| `stock_sector_fund_flow_hist` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `stock_sector_fund_flow_rank` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `stock_staq_net_stop` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `stock_sy_hy_em` | TypeError | 'NoneType' object is not subscriptable |
| `stock_sy_jz_em` | TypeError | 'NoneType' object is not subscriptable |
| `stock_us_famous_spot_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `stock_us_pink_spot_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `stock_zh_a_hist` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `stock_zh_a_hist_min_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `stock_zh_a_spot_em_async` | ServerDisconnectedError | Server disconnected |
| `stock_zh_index_daily_em` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without |
| `stock_zh_index_spot_sina` | TypeError | stock_zh_index_spot_sina() got an unexpected keyword argument 'symbol' |
| `stock_zt_pool_dtgc_em` | ValueError | 跌停股池只能获取最近 30 个交易日的数据 |
| `stock_zt_pool_zbgc_em` | ValueError | 炸板股池只能获取最近 30 个交易日的数据 |

### 跳过接口 (已知慢速/需登录)

- `amac_aoin_info` (skip_slow)
- `amac_fund_abs` (skip_slow)
- `amac_fund_account_info` (skip_slow)
- `amac_fund_info` (skip_slow)
- `amac_fund_sub_info` (skip_slow)
- `amac_futures_info` (skip_slow)
- `amac_manager_cancelled_info` (skip_slow)
- `amac_manager_classify_info` (skip_slow)
- `amac_manager_info` (skip_slow)
- `amac_member_info` (skip_slow)
- `amac_member_sub_info` (skip_slow)
- `amac_person_bond_org_list` (skip_slow)
- `amac_person_fund_org_list` (skip_slow)
- `amac_securities_info` (skip_slow)
- `macro_china_nbs_nation` (skip_params)
- `macro_china_nbs_region` (skip_params)
- `qhkc_tool_foreign` (skip_slow)
- `qhkc_tool_gdp` (skip_slow)
- `set_token` (skip_params)
- `volatility_yz_rv` (skip_params)

---

> ⚠️ 注意: 含 `_em` 后缀的接口来自东方财富，可能因反爬被封。
> 建议优先使用新浪 (`_sina`) 和同花顺 (`_ths`) 数据源。
