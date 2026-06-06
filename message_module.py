import smtplib
from email.header import Header
from email.mime.text import MIMEText
from tqdm import tqdm
import json
from sqlalchemy import text
from pandas_market_calendars import get_calendar
import pandas as pd

try:
    from database_module import get_sql_engine
except ImportError:
    get_sql_engine = None
from data_load_module import get_config


def get_stock_date(today_str, num):
    sse_calendar = get_calendar('SSE')  # 也可使用'SZSE'（深圳证券交易所）
    today = pd.Timestamp(today_str)

    previous_trading_days = sse_calendar.schedule(
    start_date=today - pd.Timedelta(days=10),  # 提前30天覆盖可能的范围
    end_date=today + pd.Timedelta(days=10),
    ).index  # 倒序排列，最近的日期在前
    if num >0 :     
        prev_trade_date = previous_trading_days[previous_trading_days > today][num-1]
    else:
        prev_trade_date = previous_trading_days[previous_trading_days < today][num]
    date_str = prev_trade_date.strftime("%Y%m%d")

    return date_str
    
def get_message():
    config  = get_config()
    today_str = '20250815'
    one_pre_day_str = get_stock_date(today_str, -1)
    two_pre_day_str = get_stock_date(today_str, -2)

    today_stocks = config['history_recommend'][today_str]
    one_pre_stocks = config['history_recommend'][one_pre_day_str]

    today_stock_data = get_stock_data(one_pre_day_str, today_stocks)
    today_stock_data = today_stock_data.loc[:,['name','stock_code', 'close', 'post_buy_price','pre_yield_rate', 'limit_times']]
    today_stock_data.columns = ['股票名称', '股票代码', '昨日收盘价', '推荐挂单价格', '昨日涨幅', '连板次数']

    
    one_pre_stock_data = get_stock_data(two_pre_day_str, one_pre_stocks)
    one_pre_stock_data = one_pre_stock_data.loc[:,['name','stock_code', 'yield_rate']]
    one_pre_stock_data.columns = ['股票名称', '股票代码', '昨日涨幅']


    
    today_stock_str = today_stock_data.to_string()
    one_pre_stock_str = one_pre_stock_data.to_string()


    # 邮件内容（确保为Unicode字符串）
    subject = "瑞瑞多因子打板一号（测试）"
    content = f'''各位有钱人好，今天是{today_str}，这里是瑞瑞每日选股模块：\n
    一、今日选股推荐\n
    {today_stock_str}\n
    二、上一日选股收益\n
    {one_pre_stock_str}\n
    '''
    print(content)
    # 创建MIMEText对象，指定纯文本格式和UTF-8编码
    msg = MIMEText(content, 'plain', 'utf-8')

    # 设置邮件头（使用Header类处理主题编码）
    msg['Subject'] = Header(subject, 'utf-8').encode()  # 编码主题
    return msg

def get_stock_data(day, stock_lst):
    if get_sql_engine is None:
        raise RuntimeError("get_sql_engine is not available; configure database_module before sending messages")
    engine = get_sql_engine('cdb')
    stock_list_str = "'" + "','".join(stock_lst) + "'"
    sql = f'''
    SELECT 
    *
      FROM (
    SELECT 
    trade_date, name,
    stock_code,
    close,
    close*1.04 as post_buy_price,
    pre_yield_rate,
    yield_rate,
    LEAD(yield_rate, 1) OVER (PARTITION BY stock_code ORDER BY trade_date) as post2_yield_rate,
    post2_close/close -1 as day2_yield_rate,
    pred_prob,limit_times,
    case 
    when post_open <= close * 1.04 then post_open
    else null
    end as buy_price,
    case 
    when post_open <= close * 1.04 then post2_close
    else null
    end  as sell_price,
    row_number() over (partition by trade_date order by pred_prob desc) as rn
    FROM pdb.stock_predict_data_close5_yield_rate
    ) AS T
    WHERE trade_date = '{day}'
    and stock_code in ({stock_list_str})
    ORDER BY TRADE_DATE DESC,rn 
    '''
    with engine.connect() as connection:
        query = text(sql)
        data = pd.read_sql(query, connection )
    return data 


def send_message():
    config  = get_config()
    send_email = config['email']['send_email']
    recieve_email = config['email']['recieve_email']
    authorization_code = config['email']['authorization_code']
    smtp = smtplib.SMTP('smtp.163.com', 25)
    smtp.login(send_email, authorization_code)
    msg = get_message()
    smtp.sendmail(send_email, recieve_email, msg.as_string())
if __name__ == '__main__':
    send_message()
