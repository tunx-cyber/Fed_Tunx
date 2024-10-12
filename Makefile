# 指定 Python 解释器（根据你的系统环境）  
PYTHON = python
  
# Python 脚本文件  
SCRIPT = main.py  
  
# 默认目标  
all: run_script  
  
# 运行 Python 脚本的目标  
run_script:  
	$(PYTHON) $(SCRIPT)