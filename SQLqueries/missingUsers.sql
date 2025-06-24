-- Row	experiment_number	experiment_group	name	count
-- 1	null	null	null	3760
-- 2	null	null	(direct)	6010
-- 3	null	null	HEX (GA) - Germany #hybridROAS	2
-- 4	null	null	HEX (GA) - Global #adROAS	2
-- 5	null	null	HEX (GA) - Globe #adROAS	448
-- 6	null	null	HEX (GA) - T1 #adROAS	23
-- 7	null	null	HEX (GA) - T1 #hybridROAS	60
-- 8	null	null	HEX (GA) - US #hybridROAS	8
-- 9	null	null	fb4a	4
-- 10	null	null	ig4a	2
-- 11	46	0	null	16678
-- 12	46	0	(direct)	7933
-- 13	46	0	AppAgg	1
-- 14	46	0	HEX (GA) - Globe #adROAS	8240
-- 15	46	0	HEX (GA) - T1 #adROAS	2395
-- 16	46	0	HEX (GA) - T1 #hybridROAS	433
-- 17	46	0	fb4a	195
-- 18	46	0	ig4a	47
 
SELECT 
  experiment_number, 
  experiment_group,
  name,
  COUNT(*) AS count
FROM 
  `hexapolis-bcb77.Events.User_events`
WHERE 
  event_date BETWEEN "2025-03-31" AND "2025-06-22" 
  AND new_event_name = "new_user"
GROUP BY 
  experiment_number, experiment_group, name
ORDER BY 
  experiment_number, experiment_group, name
