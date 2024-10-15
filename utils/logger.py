import logging  
  
# 配置日志记录  
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式  
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期格式  
    # filename='./log/log.txt'
)  
  
# 创建一个日志记录器  
logger = logging.getLogger(__name__)