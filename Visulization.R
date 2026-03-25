############################################
# PROJECT: Short version for making all plots of this paper
############################################

##############################
# 0. SETUP
##############################

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(sf)
library(tmap)

DATA_DIR   <- "data/"
OUTPUT_DIR <- "output/"

############################################
# FIGURE 1: WORLD MAP
############################################

data(World)


World <- World %>%
  mutate(
    Color_Custom = case_when(
      name %in% c("France","Italy","Czech Rep.") ~ "#8E44AD",
      name == "Oman" ~ "#D35400",
      name %in% c("Malaysia","Georgia") ~ "#4169E1",
      name %in% c("New Zealand","Papua New Guinea") ~ "#27AE60",
      name %in% c("South Africa","Mali","Senegal") ~ "#F39C12",
      name %in% c("Chile","Suriname") ~ "#E67E22",
      TRUE ~ alpha("#d3d3d3",0.8)
    )
  )

World_sf <- st_as_sf(World)

World_map <- ggplot(World_sf) +
  geom_sf(aes(fill = Color_Custom), color = "grey50", size = 0.3) +
  scale_fill_identity() +
  theme_void()

ggsave(file.path(OUTPUT_DIR,"figure_map.pdf"), World_map, width=8, height=6)

############################################
# FIGURE 2: GDP
############################################

selected_countries <- sort(c("France","Italy","Czechia","Oman","Malaysia","Georgia","New Zealand","Papua New Guinea","South Africa","Mali","Senegal","Chile","Suriname"))

selected_world <- World %>% filter(name %in% selected_countries)

gdp_figure <- ggplot(selected_world, aes(x=reorder(name,gdp_cap_est), y=gdp_cap_est, fill=Color_Custom)) +
  geom_col() + coord_flip() + scale_fill_identity() + theme_minimal()

ggsave(file.path(OUTPUT_DIR,"figure_gdp.pdf"), gdp_figure, width=5, height=3)

############################################
# FIGURE 3: URBAN PROPORTION
############################################

Robdata <- read.csv(file.path(DATA_DIR,"R_obdata_3.csv"))

proportion_data <- Robdata %>%
  group_by(Country.Name, Degree.of.Urbanisation.Level.2) %>%
  summarise(n=n(), .groups="drop") %>%
  group_by(Country.Name) %>%
  mutate(prop=n/sum(n))

proportion_figure <- ggplot(proportion_data, aes(x=Country.Name,y=prop,fill=Degree.of.Urbanisation.Level.2)) +
  geom_bar(stat="identity") + theme_minimal()

ggsave(file.path(OUTPUT_DIR,"figure_proportion.pdf"), proportion_figure)

############################################
# FIGURE 4: R_OB VS R_REF
############################################

rob_all <- bind_rows(
  read.csv(file.path(DATA_DIR,"R_obdata_1dot5.csv")) %>% mutate(R_ref=1.5),
  read.csv(file.path(DATA_DIR,"R_obdata_3.csv")) %>% mutate(R_ref=2.5),
  read.csv(file.path(DATA_DIR,"R_obdata_4.csv")) %>% mutate(R_ref=4)
)

p_rref <- ggplot(rob_all, aes(x=Country, y=R_ob, color=factor(R_ref))) +
  geom_point(alpha=0.6, size=1.4) +
  geom_hline(aes(yintercept=R_ref, color=factor(R_ref)), linetype="dashed") +
  theme_bw()

ggsave(file.path(OUTPUT_DIR,"figure_rref.pdf"), p_rref, width=7, height=3)

############################################
# FIGURE 5: DISTRIBUTION (OB vs LOCAL)
############################################

rob_clean <- rob_all %>% filter(!is.na(R_ob), !is.na(Local_R))

rob_long <- rob_clean %>% pivot_longer(c(R_ob,Local_R), names_to="Type", values_to="Value")

R_ob_density <- ggplot(rob_long) +
  geom_histogram(aes(x=Value, fill=Type), bins=30, alpha=0.5, position="identity") +
  theme_minimal()

ggsave(file.path(OUTPUT_DIR,"figure_density.pdf"), R_ob_density)

############################################
# FIGURE 6: TOTAL DISTRIBUTION
############################################

rob_long2 <- rob_clean %>% pivot_longer(c(R_ob,R_sum), names_to="Type", values_to="Value")

R_total_density <- ggplot(rob_long2) +
  geom_histogram(aes(x=Value, fill=Type), bins=30, alpha=0.5, position="identity") +
  theme_minimal()

ggsave(file.path(OUTPUT_DIR,"figure_total_density.pdf"), R_total_density)

############################################
# FIGURE 7: BIAS BOXPLOTS
############################################

rob_all <- rob_all %>% mutate(Rbias = abs(R_ob-Local_R)/R_ob)

p_bias <- ggplot(rob_all, aes(x=DEGURBA_L2,y=Rbias,fill=factor(R_ref))) + geom_boxplot() + theme_bw()

ggsave(file.path(OUTPUT_DIR,"figure_bias.pdf"), p_bias)

############################################
# FIGURE 8: MULTI-TIME ESTIMATION
############################################

multi <- read.csv(file.path(DATA_DIR,"multitime_estimation_6_region.csv"))

multi_plot <- ggplot(multi, aes(x=max_size)) +
  geom_line(aes(y=R_ob_stat)) +
  geom_line(aes(y=R_ob), linetype="dashed") +
  theme_minimal()

ggsave(file.path(OUTPUT_DIR,"figure_multi.pdf"), multi_plot)

############################################
# FIGURE 9: SINGLE-TIME ESTIMATION
############################################

single <- read.csv(file.path(DATA_DIR,"singletime_estimation_more_place.csv"))

single_plot <- ggplot(single, aes(x=Name,y=median)) + geom_point() + theme_minimal()

ggsave(file.path(OUTPUT_DIR,"figure_single.pdf"), single_plot)

############################################
# FIGURE 10: CANADA R_ob
############################################

rob_canada <- read.csv(file.path(DATA_DIR,"Rob_posterior_canada.csv"))

rob_canada <- rob_canada %>%
  mutate(generation=factor(generation, levels=c("9generation","10generation","11generation")))

p_rob <- ggplot(rob_canada, aes(x=region,y=value,fill=generation)) + geom_boxplot()

ggsave(file.path(OUTPUT_DIR,"figure_canada_Rob.pdf"), p_rob)

############################################
# FIGURE 11: CANADA R_ref
############################################

rref_canada <- read.csv(file.path(DATA_DIR,"Rref_posterior_canada.csv"))

p_rref <- ggplot(rref_canada, aes(x=generation,y=R_ref,fill=generation)) + geom_boxplot()

ggsave(file.path(OUTPUT_DIR,"figure_canada_Rref.pdf"), p_rref)

############################################
# END
############################################
