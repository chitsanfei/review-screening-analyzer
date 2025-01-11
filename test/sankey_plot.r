setwd("C:/Users/admin/Desktop/article-analyzer")

# 加载必要的包
library(networkD3)
library(dplyr)
library(readr)

# 读取数据
data <- read_csv("data/picos_analysis.csv")

# 准备节点数据
nodes <- data.frame(
  name = c(
    "Model A True", "Model A False",
    "Model B True", "Model B False",
    "Model C True", "Model C False", "Model C NA",
    "Final True", "Final False"
  ),
  group = c(
    "A True", "A False",
    "B True", "B False",
    "C True", "C False", "C NA",
    "F True", "F False"
  )
)

# 计算流向
# A -> B
a_true_b_true <- sum(data$A_Decision & data$B_Decision, na.rm = TRUE)
a_true_b_false <- sum(data$A_Decision & !data$B_Decision, na.rm = TRUE)
a_false_b_true <- sum(!data$A_Decision & data$B_Decision, na.rm = TRUE)
a_false_b_false <- sum(!data$A_Decision & !data$B_Decision, na.rm = TRUE)

# B -> C
b_true_c_true <- sum(data$B_Decision & data$C_Decision, na.rm = TRUE)
b_true_c_false <- sum(data$B_Decision & !data$C_Decision, na.rm = TRUE)
b_true_c_na <- sum(data$B_Decision & is.na(data$C_Decision), na.rm = TRUE)
b_false_c_true <- sum(!data$B_Decision & data$C_Decision, na.rm = TRUE)
b_false_c_false <- sum(!data$B_Decision & !data$C_Decision, na.rm = TRUE)
b_false_c_na <- sum(!data$B_Decision & is.na(data$C_Decision), na.rm = TRUE)

# C -> Final
c_true_final_true <- sum(data$C_Decision & data$Final_Decision, na.rm = TRUE)
c_true_final_false <- sum(data$C_Decision & !data$Final_Decision, na.rm = TRUE)
c_false_final_true <- sum(!data$C_Decision & data$Final_Decision, na.rm = TRUE)
c_false_final_false <- sum(!data$C_Decision & !data$Final_Decision, na.rm = TRUE)
c_na_final_true <- sum(is.na(data$C_Decision) & data$Final_Decision, na.rm = TRUE)
c_na_final_false <- sum(is.na(data$C_Decision) & !data$Final_Decision, na.rm = TRUE)

# 准备链接数据
links <- data.frame(
  source = c(
    # A -> B
    rep(0, 2), rep(1, 2),
    # B -> C
    rep(2, 3), rep(3, 3),
    # C -> Final
    rep(4, 2), rep(5, 2), rep(6, 2)
  ),
  target = c(
    # A -> B
    2, 3, 2, 3,
    # B -> C
    4, 5, 6, 4, 5, 6,
    # C -> Final
    7, 8, 7, 8, 7, 8
  ),
  value = c(
    # A -> B
    a_true_b_true, a_true_b_false, a_false_b_true, a_false_b_false,
    # B -> C
    b_true_c_true, b_true_c_false, b_true_c_na,
    b_false_c_true, b_false_c_false, b_false_c_na,
    # C -> Final
    c_true_final_true, c_true_final_false,
    c_false_final_true, c_false_final_false,
    c_na_final_true, c_na_final_false
  )
)

# 创建颜色向量
my_color <- paste0(
  'd3.scaleOrdinal()
    .domain(["A True", "A False",
             "B True", "B False",
             "C True", "C False", "C NA",
             "F True", "F False"])
    .range(["#fbf8cc", "#fde4cf",
            "#FFCFD2", "#F1C0E8",
            "#CFBAF0", "#A3C4F3", "#90DBF4",
            "#98F5E1", "#B9FBC0"])'
)

# 绘制桑基图
sankeyNetwork(Links = links, Nodes = nodes,
              Source = "source", Target = "target",
              Value = "value", NodeID = "name",
              NodeGroup = "group",
              sinksRight = TRUE,
              nodeWidth = 40,
              nodePadding = 20,
              colourScale = my_color,
              fontSize = 12,
              height = 500,
              width = 800)

# 保存为HTML文件
saveNetwork(sankeyNetwork(Links = links, Nodes = nodes,
                         Source = "source", Target = "target",
                         Value = "value", NodeID = "name",
                         NodeGroup = "group",
                         sinksRight = TRUE,
                         nodeWidth = 40,
                         nodePadding = 20,
                         colourScale = my_color,
                         fontSize = 12,
                         height = 500,
                         width = 800),
           "sankey_plot.html") 