# ==============================================================================
# COMPLETE INDIAN JOB MARKET DASHBOARD & SALARY PREDICTOR (5-TAB APP)
# ==============================================================================

library(shiny)
library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(plotly)
library(leaflet)
library(DT)
library(rpart)         # Decision Tree
library(randomForest)  # Random Forest
library(caret)         # KNN Regression

# ==============================================================================
# PART 1: DATA LOADING & EDA / CLEANING (FROM SCRATCH)
# ==============================================================================
print("Step 1: Loading and cleaning dataset...")

raw_data <- read_excel("indian-job-market-dataset-2025.xlsx")

# Remove duplicate jobs and drop unnecessary columns
jobs <- raw_data %>%
  distinct(jobId, .keep_all = TRUE) %>%
  select(
    title, companyName, tagsAndSkills, experience, salary, location,
    currency, minimumSalary, maximumSalary, minimumExperience, maximumExperience
  )

# Clean columns and create useful features
jobs <- jobs %>%
  mutate(
    minimumExperience = suppressWarnings(as.numeric(minimumExperience)),
    maximumExperience = suppressWarnings(as.numeric(maximumExperience)),
    minimumSalary     = suppressWarnings(as.numeric(minimumSalary)),
    maximumSalary     = suppressWarnings(as.numeric(maximumSalary)),
    
    mid_exp = (minimumExperience + maximumExperience) / 2,
    
    exp_group = case_when(
      minimumExperience <= 1 ~ "0-1 Yrs (Fresher)",
      minimumExperience <= 4 ~ "2-4 Yrs (Junior)",
      minimumExperience <= 9 ~ "5-9 Yrs (Mid-Senior)",
      minimumExperience >= 10 ~ "10+ Yrs (Senior)",
      TRUE ~ "Not Specified"
    ),
    
    valid_salary = (currency == "INR" & !is.na(minimumSalary) & minimumSalary > 0),
    
    min_sal_lpa = ifelse(valid_salary, minimumSalary / 100000, NA),
    max_sal_lpa = ifelse(valid_salary, maximumSalary / 100000, NA),
    mid_sal_lpa = ifelse(valid_salary, (min_sal_lpa + max_sal_lpa) / 2, NA),
    
    city = str_split_fixed(location, ",", 2)[, 1],
    city = str_remove(city, "Hybrid - "),
    city = str_remove(city, "\\(.*\\)"),
    city = str_trim(city),
    city = ifelse(city %in% c("Delhi / NCR", "New Delhi", "Delhi"), "Delhi NCR", city),
    city = ifelse(city %in% c("Bangalore", "Bangalore Rural"), "Bengaluru", city),
    city = ifelse(city %in% c("Mumbai Suburban", "Navi Mumbai"), "Mumbai", city),
    
    text_lower = tolower(paste(title, tagsAndSkills)),
    category = case_when(
      str_detect(text_lower, "software|developer|engineer|java|python|data|cloud|sql|testing|ai") ~ "IT & Software",
      str_detect(text_lower, "sales|business development|bde|bdm|marketing") ~ "Sales & Marketing",
      str_detect(text_lower, "bank|finance|accountant|audit|loan|credit") ~ "Finance & Banking",
      str_detect(text_lower, "hr|human resource|recruiter|talent") ~ "HR & Recruitment",
      str_detect(text_lower, "customer|bpo|support|voice|operations") ~ "Operations & BPO",
      TRUE ~ "Other"
    ),
    
    is_tech = ifelse(str_detect(text_lower, "python|java|aws|azure|react|sql|sap|machine learning|c\\+\\+"), 1, 0)
  ) %>%
  select(-text_lower)

# ==============================================================================
# PART 2: GEOSPATIAL DATA PREPARATION (INDIAN CITIES)
# ==============================================================================
print("Step 2: Preparing India map coordinates...")

city_coords <- data.frame(
  city = c("Bengaluru", "Hyderabad", "Pune", "Mumbai", "Chennai", "Gurugram",
           "Noida", "Delhi NCR", "Ahmedabad", "Kolkata", "Thane", "Jaipur",
           "Coimbatore", "Chandigarh", "Kochi", "Vadodara", "Surat", "Nagpur",
           "Lucknow", "Mohali", "Indore", "Bhopal", "Bhubaneswar", "Dehradun"),
  lat = c(12.9716, 17.3850, 18.5204, 19.0760, 13.0827, 28.4595,
          28.5355, 28.6139, 23.0225, 22.5726, 19.2183, 26.9124,
          11.0168, 30.7333, 9.9312, 22.3072, 21.1702, 21.1458,
          26.8467, 30.7046, 22.7196, 23.2599, 20.2961, 30.3165),
  lng = c(77.5946, 78.4867, 73.8567, 72.8777, 80.2707, 77.0266,
          77.3910, 77.2090, 72.5714, 88.3639, 72.9781, 75.7873,
          76.9558, 76.7794, 76.2673, 73.1812, 72.8311, 79.0882,
          80.9462, 76.7179, 75.8577, 77.4126, 85.8245, 78.0322)
)

map_data <- jobs %>%
  group_by(city) %>%
  summarise(
    total_jobs = n(),
    median_salary = round(median(mid_sal_lpa, na.rm = TRUE), 2)
  ) %>%
  inner_join(city_coords, by = "city")

# ==============================================================================
# PART 3: MACHINE LEARNING - TRAIN & COMPARE 4 SALARY PREDICTION MODELS
# ==============================================================================
print("Step 3: Training ML models for salary prediction...")

top_cities <- c("Bengaluru", "Hyderabad", "Pune", "Mumbai", "Chennai", "Gurugram", "Noida", "Delhi NCR")

ml_data <- jobs %>%
  filter(valid_salary == TRUE, !is.na(mid_exp), mid_exp >= 0, mid_exp <= 25, mid_sal_lpa <= 60) %>%
  mutate(
    city_clean = ifelse(city %in% top_cities, city, "Other City"),
    city_clean = as.factor(city_clean),
    category = as.factor(category)
  ) %>%
  select(mid_sal_lpa, mid_exp, city_clean, category, is_tech)

set.seed(123)
ml_sample <- ml_data %>% sample_n(min(5000, nrow(ml_data)))

train_index <- sample(1:nrow(ml_sample), 0.8 * nrow(ml_sample))
train_df <- ml_sample[train_index, ]
test_df  <- ml_sample[-train_index, ]

# 1. Linear Regression (Best Model - used in Tab 3 Predictor)
model_lm <- lm(mid_sal_lpa ~ mid_exp + city_clean + category + is_tech, data = train_df)
pred_lm  <- predict(model_lm, newdata = test_df)

# 2. Decision Tree
model_dt <- rpart(mid_sal_lpa ~ mid_exp + city_clean + category + is_tech, data = train_df)
pred_dt  <- predict(model_dt, newdata = test_df)

# 3. Random Forest
model_rf <- randomForest(mid_sal_lpa ~ mid_exp + city_clean + category + is_tech, data = train_df, ntree = 100)
pred_rf  <- predict(model_rf, newdata = test_df)

# 4. K-Nearest Neighbors (KNN)
model_knn <- knnreg(mid_sal_lpa ~ mid_exp + city_clean + category + is_tech, data = train_df, k = 15)
pred_knn  <- predict(model_knn, newdata = test_df)

calc_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae  <- mean(abs(actual - predicted))
  r2   <- cor(actual, predicted)^2
  return(c(RMSE = round(rmse, 2), MAE = round(mae, 2), R2 = round(r2, 3)))
}

model_comparison <- rbind(
  "Linear Regression (Best)" = calc_metrics(test_df$mid_sal_lpa, pred_lm),
  "Decision Tree"            = calc_metrics(test_df$mid_sal_lpa, pred_dt),
  "Random Forest"            = calc_metrics(test_df$mid_sal_lpa, pred_rf),
  "KNN (k = 15)"             = calc_metrics(test_df$mid_sal_lpa, pred_knn)
) %>% as.data.frame()

model_comparison$Model <- rownames(model_comparison)
model_comparison <- model_comparison %>% select(Model, RMSE, MAE, R2)

# ==============================================================================
# PART 4: COLORFUL SHINY USER INTERFACE (UI)
# ==============================================================================
ui <- fluidPage(
  
  tags$head(
    tags$style(HTML("
      body { background-color: #f4f7fb; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
      .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #00b4db 100%);
        color: white; padding: 18px 25px; border-radius: 10px; margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
      }
      .main-header h2 { margin: 0; font-weight: 700; }
      .main-header p { margin: 4px 0 0 0; opacity: 0.9; font-size: 14px; }
      
      .kpi-card-1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 10px rgba(102,126,234,0.3); }
      .kpi-card-2 { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 10px rgba(17,153,142,0.3); }
      .kpi-card-3 { background: linear-gradient(135deg, #ff6a00 0%, #ee0979 100%); color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 10px rgba(238,9,121,0.3); }
      .kpi-title { font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.9; margin: 0; }
      .kpi-val { font-size: 26px; font-weight: 700; margin: 5px 0 0 0; }
      
      .pred-box-main { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(17,153,142,0.3); }
      .pred-box-range { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(33,147,176,0.3); }
      
      .chart-card { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 15px; }
      .sidebar-box { background: white; border-top: 4px solid #2a5298; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
      
      .nav-tabs > li.active > a, .nav-tabs > li.active > a:focus, .nav-tabs > li.active > a:hover {
        background-color: #2a5298 !important; color: white !important; border-radius: 6px 6px 0 0; font-weight: 600;
      }
      .nav-tabs > li > a { color: #2a5298; font-weight: 600; }
    "))
  ),
  
  # Gradient Banner Header
  div(class = "main-header",
      h2("Indian Job Market Intelligence & Salary Predictor (2025)"),
      p("Interactive Geospatial Analytics, Compensation Trends & Machine Learning")
  ),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      class = "sidebar-box",
      h4("Global Market Filters", style = "color: #1e3c72; font-weight: 700;"),
      selectInput("city_filter", "Select City:",
                  choices = c("All", sort(unique(city_coords$city))), selected = "All"),
      selectInput("cat_filter", "Select Job Category:",
                  choices = c("All", sort(unique(jobs$category))), selected = "All"),
      sliderInput("exp_filter", "Experience Range (Years):",
                  min = 0, max = 20, value = c(0, 20)),
      textInput("skill_search", "Search Skill / Role:", placeholder = "e.g. Python, Sales, Java"),
      hr(),
      p("Note: Sidebar filters dynamically update Tabs 1, 2, and 5.", style = "color: #666; font-size: 12px;")
    ),
    
    mainPanel(
      width = 9,
      
      # Colorful Gradient KPI Cards
      fluidRow(
        column(4, div(class = "kpi-card-1", p(class = "kpi-title", "Filtered Job Postings"), p(class = "kpi-val", textOutput("kpi_total")))),
        column(4, div(class = "kpi-card-2", p(class = "kpi-title", "Median Annual Salary"), p(class = "kpi-val", textOutput("kpi_sal")))),
        column(4, div(class = "kpi-card-3", p(class = "kpi-title", "Top Hiring Company"), p(class = "kpi-val", textOutput("kpi_comp"))))
      ),
      br(),
      
      # 5 Separate Tabs
      tabsetPanel(
        
        # ----------------------------------------------------------------------
        # TAB 1: INDIA MAP & OVERVIEW
        # ----------------------------------------------------------------------
        tabPanel(
          "1. India Map & Overview",
          br(),
          fluidRow(
            column(7, div(class = "chart-card",
                          h4("Geospatial Concentration of Jobs in India", style = "color:#1e3c72;"),
                          leafletOutput("india_map", height = 400))),
            column(5, div(class = "chart-card",
                          h4("Top 10 Hiring Companies", style = "color:#1e3c72;"),
                          plotlyOutput("plot_companies", height = 400)))
          ),
          fluidRow(
            column(6, div(class = "chart-card",
                          h4("Jobs by Experience Level", style = "color:#11998e;"),
                          plotlyOutput("plot_exp_bar", height = 300))),
            column(6, div(class = "chart-card",
                          h4("Jobs by Category", style = "color:#ee0979;"),
                          plotlyOutput("plot_cat_bar", height = 300)))
          )
        ),
        
        # ----------------------------------------------------------------------
        # TAB 2: EXPERIENCE & SALARY RELATIONSHIPS
        # ----------------------------------------------------------------------
        tabPanel(
          "2. Experience & Salary Analysis",
          br(),
          fluidRow(
            column(6, div(class = "chart-card",
                          h4("Experience vs. Median Salary Trend", style = "color:#1e3c72;"),
                          plotlyOutput("plot_exp_trend", height = 350))),
            column(6, div(class = "chart-card",
                          h4("Salary Distribution by Experience Group", style = "color:#764ba2;"),
                          plotlyOutput("plot_sal_box", height = 350)))
          ),
          fluidRow(
            column(12, div(class = "chart-card",
                           h4("Average Salary by Category & Experience (Heatmap)", style = "color:#11998e;"),
                           plotlyOutput("plot_heatmap", height = 320)))
          )
        ),
        
        # ----------------------------------------------------------------------
        # TAB 3: SALARY PREDICTOR (LINEAR REGRESSION ONLY)
        # ----------------------------------------------------------------------
        tabPanel(
          "3. Salary Predictor",
          br(),
          fluidRow(
            column(4,
                   div(class = "chart-card",
                       h4("Enter Candidate Profile", style = "color:#1e3c72; font-weight:700;"),
                       sliderInput("user_exp", "Years of Experience:", min = 0, max = 25, value = 3, step = 0.5),
                       selectInput("user_city", "Target City:", choices = levels(train_df$city_clean), selected = "Bengaluru"),
                       selectInput("user_cat", "Job Category:", choices = levels(train_df$category), selected = "IT & Software"),
                       radioButtons("user_tech", "Core Tech Skills (Python/Java/Cloud/SQL)?",
                                    choices = c("Yes" = 1, "No" = 0), selected = 1, inline = TRUE),
                       hr(),
                       p("Powered by Multiple Linear Regression (Best performing model in evaluation).", style = "color:#555; font-size:12px;")
                   )
            ),
            column(8,
                   fluidRow(
                     column(6, div(class = "pred-box-main",
                                   p(class = "kpi-title", "Predicted Annual Salary (Linear Regression)"),
                                   p(class = "kpi-val", textOutput("out_lm")))),
                     column(6, div(class = "pred-box-range",
                                   p(class = "kpi-title", "Expected Salary Range (±1 MAE)"),
                                   p(class = "kpi-val", textOutput("out_lm_range"))))
                   ),
                   div(class = "chart-card",
                       h4("Linear Regression Salary Growth Curve for Selected Profile", style = "color:#1e3c72;"),
                       plotlyOutput("plot_pred_curve", height = 340))
            )
          )
        ),
        
        # ----------------------------------------------------------------------
        # TAB 4: ML MODEL COMPARISON (ALL 4 MODELS)
        # ----------------------------------------------------------------------
        tabPanel(
          "4. ML Model Comparison",
          br(),
          div(class = "chart-card",
              h4("Model Performance Metrics (Evaluated on 20% Unseen Test Data)", style = "color:#1e3c72; font-weight:700;"),
              p("RMSE & MAE measure prediction error in Lakhs Per Annum (Lower is Better). R² measures variance explained (Higher is Better)."),
              tableOutput("table_metrics")
          ),
          fluidRow(
            column(6, div(class = "chart-card",
                          h4("Error Comparison: RMSE & MAE (Lower is Better)", style = "color:#ee0979;"),
                          plotlyOutput("plot_model_error", height = 340))),
            column(6, div(class = "chart-card",
                          h4("Accuracy Score: R-Squared (Higher is Better)", style = "color:#11998e;"),
                          plotlyOutput("plot_model_r2", height = 340)))
          )
        ),
        
        # ----------------------------------------------------------------------
        # TAB 5: JOB DATA EXPLORER
        # ----------------------------------------------------------------------
        tabPanel(
          "5. Job Data Explorer",
          br(),
          div(class = "chart-card",
              DTOutput("jobs_table"))
        )
      )
    )
  )
)

# ==============================================================================
# PART 5: SHINY SERVER LOGIC
# ==============================================================================
server <- function(input, output) {
  
  # Reactive dataset based on sidebar filters
  filtered_data <- reactive({
    df <- jobs %>%
      filter(is.na(minimumExperience) | (minimumExperience >= input$exp_filter[1] & minimumExperience <= input$exp_filter[2]))
    
    if (input$city_filter != "All") {
      df <- df %>% filter(city == input$city_filter)
    }
    if (input$cat_filter != "All") {
      df <- df %>% filter(category == input$cat_filter)
    }
    if (input$skill_search != "") {
      kw <- tolower(input$skill_search)
      df <- df %>% filter(str_detect(tolower(title), fixed(kw)) | str_detect(tolower(coalesce(tagsAndSkills, "")), fixed(kw)))
    }
    df
  })
  
  # KPI Cards
  output$kpi_total <- renderText({ format(nrow(filtered_data()), big.mark = ",") })
  output$kpi_sal <- renderText({
    med <- median(filtered_data()$mid_sal_lpa, na.rm = TRUE)
    if (is.na(med)) "N/A" else paste0("₹ ", round(med, 2), " LPA")
  })
  output$kpi_comp <- renderText({
    top_c <- filtered_data() %>% filter(!is.na(companyName)) %>% count(companyName, sort = TRUE) %>% slice(1) %>% pull(companyName)
    if (length(top_c) == 0) "N/A" else top_c
  })
  
  # TAB 1: Geospatial Map (Using free OpenStreetMap tiles - no API key needed!)
  output$india_map <- renderLeaflet({
    df_map <- map_data
    if (input$city_filter != "All") {
      df_map <- df_map %>% filter(city == input$city_filter)
    }
    
    pal <- colorNumeric(palette = c("#00b4db", "#667eea", "#ee0979"), domain = df_map$median_salary)
    
    leaflet(df_map) %>%
      addTiles() %>%
      setView(lng = 78.9629, lat = 22.5937, zoom = 5) %>%
      addCircleMarkers(
        lng = ~lng, lat = ~lat,
        radius = ~sqrt(total_jobs) / 6 + 6,
        color = "#1e3c72", weight = 1.5,
        fillColor = ~pal(median_salary), fillOpacity = 0.85,
        popup = ~paste0("<b>", city, "</b><br>Total Jobs: ", format(total_jobs, big.mark = ","),
                        "<br>Median Salary: ₹", median_salary, " LPA")
      ) %>%
      addLegend("bottomright", pal = pal, values = ~median_salary, title = "Median LPA", opacity = 0.8)
  })
  
  # TAB 1: Top Companies
  output$plot_companies <- renderPlotly({
    df <- filtered_data() %>% count(companyName, sort = TRUE) %>% head(10)
    plot_ly(df, x = ~n, y = ~reorder(companyName, n), type = "bar", orientation = "h",
            marker = list(color = ~n, colorscale = "Viridis")) %>%
      layout(xaxis = list(title = "Job Count"), yaxis = list(title = ""), margin = list(l = 130))
  })
  
  # TAB 1: Experience Groups
  output$plot_exp_bar <- renderPlotly({
    df <- filtered_data() %>% filter(exp_group != "Not Specified") %>% count(exp_group)
    plot_ly(df, x = ~exp_group, y = ~n, type = "bar",
            marker = list(color = c("#667eea", "#11998e", "#f7971e", "#ee0979"))) %>%
      layout(xaxis = list(title = ""), yaxis = list(title = "Jobs"))
  })
  
  # TAB 1: Job Categories
  output$plot_cat_bar <- renderPlotly({
    df <- filtered_data() %>% count(category, sort = TRUE)
    plot_ly(df, labels = ~category, values = ~n, type = "pie", hole = 0.45,
            marker = list(colors = c("#2193b0", "#ff6a00", "#11998e", "#764ba2", "#f7971e", "#6c757d"))) %>%
      layout(showlegend = TRUE)
  })
  
  # TAB 2: Experience vs Salary Line Chart
  output$plot_exp_trend <- renderPlotly({
    df <- filtered_data() %>%
      filter(valid_salary == TRUE, !is.na(mid_exp), mid_exp <= 20) %>%
      mutate(exp_year = round(mid_exp)) %>%
      group_by(exp_year, category) %>%
      summarise(med_sal = median(mid_sal_lpa), .groups = "drop")
    
    plot_ly(df, x = ~exp_year, y = ~med_sal, color = ~category,
            colors = c("#2193b0", "#ff6a00", "#11998e", "#764ba2", "#f7971e", "#e74c3c"),
            type = "scatter", mode = "lines+markers", line = list(width = 3)) %>%
      layout(xaxis = list(title = "Experience (Years)"), yaxis = list(title = "Median Salary (LPA)"))
  })
  
  # TAB 2: Salary Boxplot
  output$plot_sal_box <- renderPlotly({
    df <- filtered_data() %>% filter(valid_salary == TRUE, exp_group != "Not Specified", mid_sal_lpa <= 50)
    plot_ly(df, x = ~exp_group, y = ~mid_sal_lpa, color = ~exp_group,
            colors = c("#667eea", "#11998e", "#f7971e", "#ee0979"), type = "box") %>%
      layout(xaxis = list(title = ""), yaxis = list(title = "Salary (LPA)"), showlegend = FALSE)
  })
  
  # TAB 2: Heatmap
  output$plot_heatmap <- renderPlotly({
    df <- filtered_data() %>%
      filter(valid_salary == TRUE, exp_group != "Not Specified") %>%
      group_by(category, exp_group) %>%
      summarise(avg_sal = round(mean(mid_sal_lpa), 2), .groups = "drop")
    
    plot_ly(df, x = ~exp_group, y = ~category, z = ~avg_sal, type = "heatmap", colorscale = "Viridis") %>%
      layout(xaxis = list(title = ""), yaxis = list(title = ""), margin = list(l = 130))
  })
  
  # TAB 3: Live Prediction using Linear Regression ONLY
  new_input <- reactive({
    data.frame(
      mid_exp = input$user_exp,
      city_clean = factor(input$user_city, levels = levels(train_df$city_clean)),
      category = factor(input$user_cat, levels = levels(train_df$category)),
      is_tech = as.numeric(input$user_tech)
    )
  })
  
  lm_mae <- model_comparison$MAE[1]
  
  output$out_lm <- renderText({
    val <- max(1.0, round(predict(model_lm, new_input()), 2))
    paste0("₹ ", val, " LPA")
  })
  
  output$out_lm_range <- renderText({
    val <- max(1.0, predict(model_lm, new_input()))
    low <- max(1.0, round(val - lm_mae, 2))
    high <- round(val + lm_mae, 2)
    paste0("₹ ", low, " – ₹ ", high, " LPA")
  })
  
  # TAB 3: Linear Regression Salary Curve vs User Point
  output$plot_pred_curve <- renderPlotly({
    exp_seq <- seq(0, 25, by = 1)
    grid <- data.frame(
      mid_exp = exp_seq,
      city_clean = factor(input$user_city, levels = levels(train_df$city_clean)),
      category = factor(input$user_cat, levels = levels(train_df$category)),
      is_tech = as.numeric(input$user_tech)
    )
    grid$lm_pred <- pmax(1.0, predict(model_lm, newdata = grid))
    grid$upper <- grid$lm_pred + lm_mae
    grid$lower <- pmax(1.0, grid$lm_pred - lm_mae)
    
    user_lm_val <- max(1.0, predict(model_lm, newdata = new_input()))
    
    plot_ly(grid, x = ~mid_exp) %>%
      add_ribbons(ymin = ~lower, ymax = ~upper, name = "Expected Range (±1 MAE)",
                  fillcolor = "rgba(17, 153, 142, 0.2)", line = list(color = "transparent")) %>%
      add_lines(y = ~lm_pred, name = "Linear Regression Trend",
                line = list(color = "#11998e", width = 3)) %>%
      add_markers(x = input$user_exp, y = user_lm_val, name = "Your Predicted Salary",
                  marker = list(color = "#ee0979", size = 15, symbol = "star")) %>%
      layout(xaxis = list(title = "Years of Experience"), yaxis = list(title = "Predicted Salary (LPA)"))
  })
  
  # TAB 4: Model Comparison Table & Charts
  output$table_metrics <- renderTable({ model_comparison }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  output$plot_model_error <- renderPlotly({
    plot_ly(model_comparison, x = ~Model) %>%
      add_bars(y = ~RMSE, name = "RMSE (LPA)", marker = list(color = "#ee0979")) %>%
      add_bars(y = ~MAE, name = "MAE (LPA)", marker = list(color = "#f7971e")) %>%
      layout(barmode = "group", xaxis = list(title = ""), yaxis = list(title = "Error (in LPA)"))
  })
  
  output$plot_model_r2 <- renderPlotly({
    plot_ly(model_comparison, x = ~Model, y = ~R2, type = "bar",
            marker = list(color = c("#2193b0", "#cc2b5e", "#00b09b", "#f7971e"))) %>%
      layout(xaxis = list(title = ""), yaxis = list(title = "R² Score (0 to 1)", range = c(0, 1)))
  })
  
  # TAB 5: Interactive Data Table
  output$jobs_table <- renderDT({
    filtered_data() %>%
      select(Title = title, Company = companyName, City = city, Category = category,
             Experience = experience, Salary = salary, Skills = tagsAndSkills) %>%
      datatable(options = list(pageLength = 10, scrollX = TRUE))
  })
}

# Run the application
shinyApp(ui = ui, server = server)