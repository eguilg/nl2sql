# -*- coding: utf-8 -*-

import re




def strPreProcess(question):
    value = question
    try:
        if re.search(r'为负值|为负', value):
            value = re.sub(r'为负值|为负', '小于0', value)
        if re.search(r'为正值|为正', value):
            value = re.sub(r'为正值|为正', '大于0', value)
        # X.x块钱  X毛钱
        value = value.replace('块钱', '元')
        patten_money = re.compile(r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}点[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}')
        k = patten_money.findall(value)
        if k:
            for item in k:
                listm = item.split('点')
                front, rf = chinese_to_digits(listm[0])
                end, rn = chinese_to_digits(listm[1])
                val = str(front) + '.' + str(end)
                value = value.replace(item, val, 1)
        patten_kuai = re.compile(r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}块[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{,1}')
        km = patten_kuai.findall(value)
        if km:
            for item in km:
                listm = item.split('块')
                front, rf = chinese_to_digits(listm[0])
                end, rn = chinese_to_digits(listm[1])
                if end:
                    val = str(front) + '.' + str(end) + '元'
                else:
                    val = str(front) + '元'
                value = value.replace(item, val, 1)
            # value = value.replace('毛钱', '元',)
            # value = value.replace('毛', '元')
        patten_mao = re.compile(r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1}毛|[0-9]毛')
        kmao = patten_mao.findall(value)
        if kmao:
            for item in kmao:
                strmao = item.replace('毛', '')
                valmao, rm = chinese_to_digits(strmao)
                maoflo = str(float(valmao)/10) + '元'
                value = value.replace(item, maoflo, 1)
        value = value.replace('元毛', '元')
        mm = re.findall(r'[〇|零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{2,}',value)
        if mm:
            for item in mm:
                v, r = chinese_to_digits(item)
                if r ==1 and v//10 + 1 !=len(item):
                    v = str(v).zfill(len(item) - v//10)
                value = value.replace(item, str(v),1)

        #百分之几----\d%
        if re.search(r'百分之', value):
            items = re.findall(r'百分之[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}', value)
            #items= re.findall(r'百分之\d*?}', value)
            if items:
                for item in items:
                    item_t = item.replace('百分之', '')
                    k, r = chinese_to_digits(item_t)
                    item_t = str(k) + '%'
                    value = re.sub(str(item), str(item_t), value)
                    # print('1--',items,value)
            items_two = re.findall(r'百分之\d{1,}\.?\d*', value)
            if items_two:
                for item in items_two:
                    item_t = item.replace('百分之', '') + '%'
                    value = re.sub(str(item), str(item_t), value)
                    # print('2--', items_two, value)
        if re.search(r'百分点', value):
            items_we = re.findall(r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}.??百分点', value)
            if items_we:
                for item in items_we:
                    item_t = re.sub('.??百分点', '', item)
                    k,r = chinese_to_digits(item_t)
                    item_t = str(k) + '%'
                    value = re.sub(str(item), str(item_t), value)
                # print('百分点-中',items_we,value)
            items_se = re.findall(r'\d+?\.??\d*.??百分点', value)
            if items_se:
                for item in items_se:
                    item_t = re.sub('.??百分点', '', item) + '%'
                    value = re.sub(str(item), str(item_t), value)
                # print('百分点-ala', items_se, value)
        # 更改中文年份并补充完整
        pattern_date1 = re.compile(r'(\d{2,4}年)')
        #pattern_date1 = re.compile(r'(.{1}月.{,2})日|号')
        date1 = pattern_date1.findall(value)
        dateList1 = list(set(date1))
        if dateList1:
            for item in dateList1:
                v = str_to_date(item)
                value = re.sub(str(item), str(v), value)

        pattern_date2 = re.compile(r'(\d+)(\-|\.)(\d+)(\-|\.)(\d+)')
        date2 = pattern_date2.findall(value)
        dateList2 = list(set(date2))
        if dateList2:
            for item in dateList2:
                v = str_to_date(item)
                value = re.sub(str(item), str(v), value)
        pattern_date3 = re.compile(r'(.{1})月(.{1})(号|日)')
        date3 = pattern_date3.findall(value)
        if date3:
            for item in date3:
                for i in range(len(item) -1):
                    if item[i].isdigit():
                        pass
                    else:
                        k,r = chinese_to_digits(item[i])
                        value = value.replace(item[i],str(k),1)
        pattern_date4 = re.compile(r'\d*?年[\D]{1}月')
        date4 = pattern_date4.findall(value)
        if date4:
            for item in date4:
                kitem = re.findall(r'([\D]{1})月',item)
                k,v = chinese_to_digits(kitem[0])
                mm = item.replace(kitem[0],str(k))
                value = re.sub(item, mm, value)



    except Exception as exc:
        print(exc)

    return value


# 汉字数字转阿拉伯数字
def chinese_to_digits(uchars_chinese):
    total = 0

    common_used_numerals_tmp = {
        '〇': 0,
        '零': 0,
        '一': 1,
        '幺': 1,
        '二': 2,
        '两': 2,
        '三': 3,
        '四': 4,
        '五': 5,
        '六': 6,
        '七': 7,
        '八': 8,
        '九': 9,
        '十': 10,
        '百': 100
    }
    r = 1  # 表示单位：个十百千...
    try:

        for i in range(len(uchars_chinese) - 1, -1, -1):
            # print(uchars_chinese[i])
            val = common_used_numerals_tmp.get(uchars_chinese[i])

            if val is not None:
                # print('val', val)
                if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
                    if val > r:
                        r = val
                        total = total + val
                    else:
                        r = r * val
                        # total = total + r * x
                elif val >= 10:
                    if val > r:
                        r = val
                    else:
                        r = r * val
                elif r == 1:
                    total = total + pow(10,len(uchars_chinese) - i - 1) * val
                else:
                    total = total + r * val
    except Exception as exc:
        print(uchars_chinese)
        print(exc)
    return total, r


# 日期字符转日期
def str_to_date(date_str):
    try:
        # 是数字 有年月日三位
        date_search = re.search('(\d+)(\-|\.)(\d+)(\-|\.)(\d+)', date_str)
        if date_search:
            year_str = date_search.group(1)
            month_str = date_search.group(3)
            day_str = date_search.group(5)
            if len(year_str) == 2:
                year_str = '20' + year_str
            if len(year_str) == 3:
                year_str = '2' + year_str
            date_date = '{}年{}月{}日'.format(year_str, month_str, day_str)
            return date_date

        # 是数字 只有年月
        # 辅导公告 默认是月底
        date_search = re.search('(\d+)(\-|\.)(\d+)', date_str)
        if date_search:
            year_str = date_search.group(1)
            month_str = date_search.group(3)
            if len(year_str) == 2:
                year_str = '20' + year_str
            if len(year_str) == 3:
                year_str = '2' + year_str
            date_date = '%s年%s月' % (year_str, month_str)
            return date_date

        # 以下包含汉字
        date_str = date_str.replace('号', '日')
        # 有年月日三位
        date_search = re.search('(.{2,4})年(.*?)月(.*?)日', date_str)
        if date_search:
            if date_search.group(1).isdigit():  # 不能用isnumeric 汉字一二三四会被认为是数字
                # 只有年月日是汉字 数字还是阿拉伯数字
                year_str = date_search.group(1)
                month_str = date_search.group(2)
                day_str = date_search.group(3)
            # 年份不足4位 把前面的补上
            if len(year_str) == 2:
                year_str = '20' + year_str
            if len(year_str) == 3:
                year_str = '2' + year_str
            date_str = '%s年%s月%s日' % (year_str, month_str, day_str)
            return date_str

        # 只有两位
        date_search = re.search('(.{2,4})年(.*?)月', date_str)
        if date_search:
            if date_search.group(1).isdigit():
                year_str = date_search.group(1)
                month_str = date_search.group(2)
            if len(year_str) == 2:
                year_str = '20' + year_str
            if len(year_str) == 3:
                year_str = '2' + year_str
            date_str = '%s年%s月' % (year_str, month_str)
            return date_str
        # 只有一位

        date_search = re.search('(\d{2,4})年', date_str)
        if date_search:
            if date_search.group(1).isdigit():
                year_str = date_search.group(1)
            if len(year_str) == 2 and int(year_str[0]) < 2:
                year_str = '20' + year_str
            if len(year_str) == 3:
                year_str = '2' + year_str
            date_str = '%s年' % (year_str)
            return date_str

        # print('处理不了的日期 %s' % date_str)
    except Exception as exc:
        pass

    return None


str_test1 = '11和2012年,19年1月7日到十九日周票房超过一千万的影投公司,幺九年一月十四到十九播放数大于三千万的剧集,18年同期'
str_test2 = '市值是不超过百亿元,股价高于十块钱,增长超过两块五,或者上涨幅度大于百分之八的股票'
str_test3 = '涨幅为正，年收益为正值，税后利率不为负，税后利润不为负值的股票'
str_test4 = '2019年第1周超过一千万并且占比高于百分之十的,百分之16，百分之几,百分之92.5,百分之0.2，十五个百分点，八个百分点'
str_test5 = '请问有哪些综艺节目它的收视率超过百分之0.2或者市场的份额超过百分之2的'
str_test6 = '中国国航的指标是什么啊，它的油价汇率不足3.5个百分点'
str_test7 = '你知道零七年五月三号,一五年七月，分别领人名币三千块钱，改革开放三十年,给你十块'
str_test8 = '三块五毛钱，四块五毛钱，三千万，六点五块钱，八点五块钱，五毛钱'
'''
你好啊请问一下上海哪些楼盘的价格在2012年的五月份超过了一万五一平-----你好啊请问一下上海哪些楼盘的价格在2012年的五月份超过了一万51平
请问一下有没有什么股票交易交割高于七块钱一股的-----请问一下有没有什么股票交易交割高于七元一股的
二月四号到十号，排名前十的院线有几个总票房大于四亿的-----2月4号到十号，排名前十的院线有几个总票房大于四亿的
保利地产公司股11年每股盈余超过一元，那它12年的每股盈余又会是多少呀-----保利地产公司股2011年每股盈余超过一元，那它2012年的每股盈余又会是多少呀
想知道有多少家影投公司第四周的票房是超过一千五百万？-----想知道有多少家影投公司第四周的票房是超过一千500万？
我想咨询一下有哪些地产股票股价是不低于十块而且在11年每股税后利润还高于一块一股-----我想咨询一下有哪些地产股票股价是不低于十块而且在2011年每股税后利润还高于1.1元股
贷款年限10年假设降20个基点调整前后是什么情况才能使每月减少还款不足100元-----贷款年限2010年假设降20个基点调整前后是什么情况才能使每月减少还款不足100元

'''

#patten = re.compile(r'[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}块[零|一|幺|二|两|三|四|五|六|七|八|九|十|百]{1,}')

# with open("F:\\天池比赛\\nl2sql_train_20190618\\train.json", "r", encoding='utf-8') as fr,open("F:\\天池比赛\\nl2sql_train_20190618\\log.txt", "w+", encoding='utf-8') as fw:
#     count = 0
#     for line in fr.readlines():
#         lines = eval(line)
#         value_re = strPreProcess(lines['question'])
#         count += 1
#         if value_re != lines['question']:
#             string = lines['question'] + '-----' + value_re + '\n'
#             fw.write(str(string))
#     print('count',count)


# value_re = strPreProcess(str_test7)
# print('----',value_re)



'''
        if re.search(r'1下|1共|1句|1线|哪1年|哪1天', value):
            value = value.replace('1下', '一下')
            value = value.replace('1句', '一句')
            value = value.replace('1共', '一共')
            value = value.replace('1线', '一线')
            value = value.replace('哪1年', '哪一年')
            value = value.replace('哪1天', '哪一天')
        if re.search(r'1手房|2手房|2线|2办', value):
            value = value.replace('1手房', '一手房')
            value = value.replace('2手房', '二手房')
            value = value.replace('2线', '二线')
            value = value.replace('2办', '两办')
'''