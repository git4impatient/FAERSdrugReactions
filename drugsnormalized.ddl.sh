#6668400|2|0.541586|*AMLODIPINE BESYLATE AND VALSARTAN*EXFORGE


impala-shell <<eoj
drop table if exists drugsnormalized;
use fda;

drop table if exists drugsnormalized;
create external table drugsnormalized ( 
isr string ,
medcount int,
druglisthash float,
druglist string
)
row format delimited fields terminated by '|'
location '/user/marty/fdastage/drugsnormalized'
;
select count(*) from  drugsnormalized;
select * from drugsnormalized limit 5;
drop table if exists drugnormed_p;

create table drugnormed_p stored as parquet as select * from drugsnormalized;
compute stats drugnormed_p;

eoj
