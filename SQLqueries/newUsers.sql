-- Warning: experiment_number and experiment_group may be null!

SELECT
  user_pseudo_id, 
  experiment_number, 
  experiment_group,
  name,
  medium,
  country,
FROM 
  `hexapolis-bcb77.Events.User_events`
WHERE 
  event_date BETWEEN "2025-03-31" AND "2025-06-22" 
  AND new_event_name = "new_user"
