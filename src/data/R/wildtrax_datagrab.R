# This is an R sctipt to:
# auth into wildtrax
# get a list of all the projects that are public or BU
# get the location data for each project
# merge the location data into one dataframe


library(wildRtrax)
library(tidyverse) # to use filter()
library(lares) # for auth credentials

### Auth into wildtrax
# save login credentials somewhere else. Followed this guide:
# https://datascienceplus.com/how-to-manage-credentials-and-secrets-safely-in-r/
get_creds()

Sys.setenv(WT_USERNAME = get_creds()$service1$username)
Sys.setenv(WT_PASSWORD = get_creds()$service1$password)

wt_auth()
wt_get_download_summary(sensor_id = "ARU")

# Download a report of the available ARU data and filter by public 
dlreport <- wt_get_download_summary(sensor_id = "ARU")
report_df <- data.frame(dlreport)
public__and_bu_project_ids <- report_df %>%
  filter(status == "Published - Public" | organization == "BU") %>%
  select(project_id)

public__and_bu_project_ids

public__and_bu_project_ids[, 1]

# get the names of the columns from the dataframe

column_names <- names(wt_download_report(public__and_bu_project_ids[, 1][1], sensor_id = 'ARU', reports='location', weather_cols = 'False'))

# make a new, empty dataframe called locations with the same column names

column_names <- c(column_names)


locations <- data.frame(matrix(ncol = length(column_names), nrow = 0))
colnames(locations) <- column_names

# Download the location data for each project and merge into one dataframe


# This part doesn't work yet - overwriting locations on each iteration of the for loop. 
for (project in public__and_bu_project_ids[,1])
{
df2 <- wt_download_report(project, sensor_id = 'ARU', reports='location', weather_cols = 'False')
df2 <- df2[names(locations)]
locations <- rbind(locations, df2)
}

save(locations, file = "locations.RData")

locations
