#ISR$OUTC_COD
#6668400$OT$
#6668640$DS$
#6668731$HO$

impala-shell <<eoj
use fda;

drop table if exists outcomestage;
create external table outcomestage ( 
isr string ,
outcome string,
dummy string
)
row format delimited fields terminated by '$'
location '/user/marty/fdastage/outcome'
;
select count(*) from  outcomestage;
select * from outcomestage limit 5;
create table outcome_p stored as parquet as select * from outcomestage;
compute stats outcome_p;

eoj
