# ==============================================================================
# COMPLETE INDIAN JOB MARKET DASHBOARD & SALARY PREDICTOR (SINGLE FILE)
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
    # Force experience & salary columns to numeric first
    minimumExperience = suppressWarnings(as.numeric(minimumExperience)),
    maximumExperience = suppressWarnings(as.numeric(maximumExperience)),
    minimumSalary     = suppressWarnings(as.numeric(minimumSalary)),
    maximumSalary     = suppressWarnings(as.numeric(maximumSalary)),
    
    # Average experience
    mid_exp = (minimumExperience + maximumExperience) / 2,
    
    # Experience groups
    exp_group = case_when(
      minimumExperience <= 1 ~ "0-1 Yrs (Fresher)",
      minimumExperience <= 4 ~ "2-4 Yrs (Junior)",
      minimumExperience <= 9 ~ "5-9 Yrs (Mid-Senior)",
      minimumExperience >= 10 ~ "10+ Yrs (Senior)",
      TRUE ~ "Not Specified"
    ),
    
    # Valid salary flag (INR only, and salary > 0 because 0 means 'Not disclosed')
    valid_salary = (currency == "INR" & !is.na(minimumSalary) & minimumSalary > 0),
    
    # Convert salary from Rupees to Lakhs Per Annum (LPA)
    min_sal_lpa = ifelse(valid_salary, minimumSalary / 100000, NA),
    max_sal_lpa = ifelse(valid_salary, maximumSalary / 100000, NA),
    mid_sal_lpa = ifelse(valid_salary, (min_sal_lpa + max_sal_lpa) / 2, NA),
    
    # Clean primary city name (take first city before comma, remove extra brackets)
    city = str_split_fixed(location, ",", 2)[, 1],
    city = str_remove(city, "Hybrid - "),
    city = str_remove(city, "\\(.*\\)"),
    city = str_trim(city),
    city = ifelse(city %in% c("Delhi / NCR", "New Delhi", "Delhi"), "Delhi NCR", city),
    city = ifelse(city %in% c("Bangalore", "Bangalore Rural"), "Bengaluru", city),
    city = ifelse(city %in% c("Mumbai Suburban", "Navi Mumbai"), "Mumbai", city),
    
    # Simple Job Category based on keywords in title
    text_lower = tolower(paste(title, tagsAndSkills)),
    category = case_when(
      str_detect(text_lower, "software|developer|engineer|java|python|data|cloud|sql|testing|ai") ~ "IT & Software",
      str_detect(text_lower, "sales|business development|bde|bdm|marketing") ~ "Sales & Marketing",
      str_detect(text_lower, "bank|finance|accountant|audit|loan|credit") ~ "Finance & Banking",
      str_detect(text_lower, "hr|human resource|recruiter|talent") ~ "HR & Recruitment",
      str_detect(text_lower, "customer|bpo|support|voice|operations") ~ "Operations & BPO",
      TRUE ~ "Other"
    ),
    
    # Check if job asks for tech skills (1 = Yes, 0 = No)
    is_tech = ifelse(str_detect(text_lower, "python|java|aws|azure|react|sql|sap|machine learning|c\\+\\+"), 1, 0)
  ) %>%
  select(-text_lower) # Drop temporary helper column

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
print("Step 3: Training 4 ML models for salary prediction...")

top_cities <- c("Bengaluru", "Hyderabad", "Pune", "Mumbai", "Chennai", "Gurugram", "Noida", "Delhi NCR")

# Prepare clean ML dataset (filter out extreme outliers > 60 LPA)
ml_data <- jobs %>%
  filter(valid_salary == TRUE, !is.na(mid_exp), mid_exp >= 0, mid_exp <= 25, mid_sal_lpa <= 60) %>%
  mutate(
    city_clean = ifelse(city %in% top_cities, city, "Other City"),
    city_clean = as.factor(city_clean),
    category = as.factor(category)
  ) %>%
  select(mid_sal_lpa, mid_exp, city_clean, category, is_tech)

# Take a 5,000-row random sample so Random Forest & KNN train quickly
set.seed(123)
ml_sample <- ml_data %>% sample_n(min(5000, nrow(ml_data)))

# 80% Training, 20% Testing split
train_index <- sample(1:nrow(ml_sample), 0.8 * nrow(ml_sample))
train_df <- ml_sample[train_index, ]
test_df  <- ml_sample[-train_index, ]

# --- MODEL 1: Linear Regression ---
model_lm <- lm(mid_sal_lpa ~ mid_exp + city_clean + category + is_tech, data = train_df)
pred_lm <- predict(model_lm, newdata = test_df)

# --- MODEL 2: Decision Tree (rpart) ---
model_dt <- rpart(mid_sal_lpa ~ mid_exp + city_clean + category + is_tech, data = train_df)
pred_dt <- predict(model_dt, newdata = test_df)

# --- MODEL 3: Random Forest ---
model_rf <- randomForest(mid_sal_lpa ~ mid_exp + city_clean + category + is_tech, data = train_df, ntree = 100)
pred_rf <- predict(model_rf, newdata = test_df)

# --- MODEL 4: K-Nearest Neighbors (KNN Regression, k = 15) ---
model_knn <- knnreg(mid_sal_lpa ~ mid_exp + city_clean + category + is_tech, data = train_df, k = 15)
pred_knn <- predict(model_knn, newdata = test_df)

# Helper function to calculate RMSE, MAE, and R-Squared
calc_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae  <- mean(abs(actual - predicted))
  r2   <- cor(actual, predicted)^2
  return(c(RMSE = round(rmse, 2), MAE = round(mae, 2), R2 = round(r2, 3)))
}

# Create comparison table of all 4 models
model_comparison <- rbind(
  "1. Linear Regression" = calc_metrics(test_df$mid_sal_lpa, pred_lm),
  "2. Decision Tree"     = calc_metrics(test_df$mid_sal_lpa, pred_dt),
  "3. Random Forest"     = calc_metrics(test_df$mid_sal_lpa, pred_rf),
  "4. KNN (k = 15)"      = calc_metrics(test_df$mid_sal_lpa, pred_knn)
) %>% as.data.frame()

model_comparison$Model <- rownames(model_comparison)
model_comparison <- model_comparison %>% select(Model, RMSE, MAE, R2)

print("Model Comparison Results:")
print(model_comparison)

# ==============================================================================
# PART 4: SHINY USER INTERFACE (UI)
# ==============================================================================
ui <- fluidPage(
  titlePanel("Indian Job Market Analysis & Salary Predictor (2025)"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      h4("Filter Dashboard Data"),
      selectInput("city_filter", "Select City:",
                  choices = c("All", sort(unique(city_coords$city))), selected = "All"),
      selectInput("cat_filter", "Select Job Category:",
                  choices = c("All", sort(unique(jobs$category))), selected = "All"),
      sliderInput("exp_filter", "Experience Range (Years):",
                  min = 0, max = 20, value = c(0, 20)),
      textInput("skill_search", "Search Skill / Role:", placeholder = "e.g. Python, Sales"),
      hr(),
      p("Note: Filters apply to Tabs 1, 2, and 4.")
    ),
    
    mainPanel(
      width = 9,
      
      # Top Summary Cards
      fluidRow(
        column(4, wellPanel(h4("Total Jobs"), h3(textOutput("kpi_total")))),
        column(4, wellPanel(h4("Median Salary"), h3(textOutput("kpi_sal")))),
        column(4, wellPanel(h4("Top Hiring Company"), h3(textOutput("kpi_comp"))))
      ),
      
      tabsetPanel(
        # TAB 1: Overview & Geospatial Map
        tabPanel(
          "1. India Map & Overview",
          br(),
          fluidRow(
            column(7, h4("Geospatial Concentration of Jobs in India"), leafletOutput("india_map", height = 400)),
            column(5, h4("Top 10 Hiring Companies"), plotlyOutput("plot_companies", height = 400))
          ),
          br(),
          fluidRow(
            column(6, h4("Jobs by Experience Level"), plotlyOutput("plot_exp_bar", height = 300)),
            column(6, h4("Jobs by Category"), plotlyOutput("plot_cat_bar", height = 300))
          )
        ),
        
        # TAB 2: Experience vs Salary & Relationships
        tabPanel(
          "2. Experience & Salary Analysis",
          br(),
          fluidRow(
            column(6, h4("Experience vs. Median Salary Trend"), plotlyOutput("plot_exp_trend", height = 350)),
            column(6, h4("Salary Distribution by Experience Group"), plotlyOutput("plot_sal_box", height = 350))
          ),
          br(),
          fluidRow(
            column(12, h4("Average Salary by Category & Experience (Heatmap)"), plotlyOutput("plot_heatmap", height = 320))
          )
        ),
        
        # TAB 3: ML Salary Predictor & Model Comparison
        tabPanel(
          "3. ML Salary Predictor (4 Models)",
          br(),
          fluidRow(
            column(4,
                   wellPanel(
                     h4("Predict Your Salary"),
                     sliderInput("user_exp", "Years of Experience:", min = 0, max = 25, value = 3, step = 0.5),
                     selectInput("user_city", "City:", choices = levels(train_df$city_clean), selected = "Bengaluru"),
                     selectInput("user_cat", "Job Category:", choices = levels(train_df$category), selected = "IT & Software"),
                     radioButtons("user_tech", "Do you have Tech Skills (Python/Java/Cloud/SQL)?",
                                  choices = c("Yes" = 1, "No" = 0), selected = 1)
                   )
            ),
            column(8,
                   h4("Predicted Salary Across All 4 Models (in INR LPA)"),
                   fluidRow(
                     column(6, wellPanel(h5("Linear Regression"), h3(textOutput("out_lm")))),
                     column(6, wellPanel(h5("Decision Tree"), h3(textOutput("out_dt"))))
                   ),
                   fluidRow(
                     column(6, wellPanel(h5("Random Forest"), h3(textOutput("out_rf")))),
                     column(6, wellPanel(h5("K-Nearest Neighbors (KNN)"), h3(textOutput("out_knn"))))
                   ),
                   h4("Model Accuracy Comparison (Evaluated on 20% Test Data)"),
                   p("Lower RMSE/MAE and higher R² indicate better model accuracy."),
                   tableOutput("table_metrics"),
                   plotlyOutput("plot_model_rmse", height = 250)
            )
          )
        ),
        
        # TAB 4: Raw Data Explorer
        tabPanel(
          "4. Job Data Explorer",
          br(),
          DTOutput("jobs_table")
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
  
  # TAB 1: Geospatial Map
  output$india_map <- renderLeaflet({
    df_map <- map_data
    if (input$city_filter != "All") {
      df_map <- df_map %>% filter(city == input$city_filter)
    }
    
    leaflet(df_map) %>%
      addTiles() %>%
      setView(lng = 78.9629, lat = 22.5937, zoom = 5) %>%
      addCircleMarkers(
        lng = ~lng, lat = ~lat,
        radius = ~sqrt(total_jobs) / 6 + 5,
        color = "#1B4F72", fillColor = "#2E86C1", fillOpacity = 0.7,
        popup = ~paste0("<b>", city, "</b><br>Total Jobs: ", total_jobs, "<br>Median Salary: ₹", median_salary, " LPA")
      )
  })
  
  # TAB 1: Top Companies
  output$plot_companies <- renderPlotly({
    df <- filtered_data() %>% count(companyName, sort = TRUE) %>% head(10)
    plot_ly(df, x = ~n, y = ~reorder(companyName, n), type = "bar", orientation = "h") %>%
      layout(xaxis = list(title = "Job Count"), yaxis = list(title = ""))
  })
  
  # TAB 1: Experience Groups
  output$plot_exp_bar <- renderPlotly({
    df <- filtered_data() %>% filter(exp_group != "Not Specified") %>% count(exp_group)
    plot_ly(df, x = ~exp_group, y = ~n, type = "bar", marker = list(color = "#117A65")) %>%
      layout(xaxis = list(title = ""), yaxis = list(title = "Jobs"))
  })
  
  # TAB 1: Job Categories
  output$plot_cat_bar <- renderPlotly({
    df <- filtered_data() %>% count(category, sort = TRUE)
    plot_ly(df, x = ~category, y = ~n, type = "bar", marker = list(color = "#D68910")) %>%
      layout(xaxis = list(title = ""), yaxis = list(title = "Jobs"))
  })
  
  # TAB 2: Experience vs Salary Line Chart
  output$plot_exp_trend <- renderPlotly({
    df <- filtered_data() %>%
      filter(valid_salary == TRUE, !is.na(mid_exp), mid_exp <= 20) %>%
      mutate(exp_year = round(mid_exp)) %>%
      group_by(exp_year, category) %>%
      summarise(med_sal = median(mid_sal_lpa), .groups = "drop")
    
    plot_ly(df, x = ~exp_year, y = ~med_sal, color = ~category, type = "scatter", mode = "lines+markers") %>%
      layout(xaxis = list(title = "Experience (Years)"), yaxis = list(title = "Median Salary (LPA)"))
  })
  
  # TAB 2: Salary Boxplot
  output$plot_sal_box <- renderPlotly({
    df <- filtered_data() %>% filter(valid_salary == TRUE, exp_group != "Not Specified", mid_sal_lpa <= 50)
    plot_ly(df, x = ~exp_group, y = ~mid_sal_lpa, color = ~exp_group, type = "box") %>%
      layout(xaxis = list(title = ""), yaxis = list(title = "Salary (LPA)"), showlegend = FALSE)
  })
  
  # TAB 2: Heatmap
  output$plot_heatmap <- renderPlotly({
    df <- filtered_data() %>%
      filter(valid_salary == TRUE, exp_group != "Not Specified") %>%
      group_by(category, exp_group) %>%
      summarise(avg_sal = round(mean(mid_sal_lpa), 2), .groups = "drop")
    
    plot_ly(df, x = ~exp_group, y = ~category, z = ~avg_sal, type = "heatmap", colors = "Blues") %>%
      layout(xaxis = list(title = ""), yaxis = list(title = ""))
  })
  
  # TAB 3: Live Predictions for all 4 ML models
  new_input <- reactive({
    data.frame(
      mid_exp = input$user_exp,
      city_clean = factor(input$user_city, levels = levels(train_df$city_clean)),
      category = factor(input$user_cat, levels = levels(train_df$category)),
      is_tech = as.numeric(input$user_tech)
    )
  })
  
  output$out_lm  <- renderText({ paste0("₹ ", round(predict(model_lm,  new_input()), 2), " LPA") })
  output$out_dt  <- renderText({ paste0("₹ ", round(predict(model_dt,  new_input()), 2), " LPA") })
  output$out_rf  <- renderText({ paste0("₹ ", round(predict(model_rf,  new_input()), 2), " LPA") })
  output$out_knn <- renderText({ paste0("₹ ", round(predict(model_knn, new_input()), 2), " LPA") })
  
  # TAB 3: Model Comparison Table & Bar Chart
  output$table_metrics <- renderTable({ model_comparison })
  
  output$plot_model_rmse <- renderPlotly({
    plot_ly(model_comparison, x = ~Model, y = ~RMSE, type = "bar", marker = list(color = "#C0392B")) %>%
      layout(title = "Prediction Error Comparison (RMSE - Lower is Better)",
             xaxis = list(title = ""), yaxis = list(title = "RMSE (in LPA)"))
  })
  
  # TAB 4: Interactive Data Table
  output$jobs_table <- renderDT({
    filtered_data() %>%
      select(Title = title, Company = companyName, City = city, Category = category,
             Experience = experience, Salary = salary, Skills = tagsAndSkills) %>%
      datatable(options = list(pageLength = 10, scrollX = TRUE))
  })
}

# Run the application
shinyApp(ui = ui, server = server)