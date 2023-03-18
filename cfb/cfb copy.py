# analyze the conference numbers. Am I better at predicting conference
# add in conference and attendance to model
# Add in points for and points against by opp by week


import pandas as pd
import pandasql as ps
import cfbd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class cfb:
    def __init__(self):
        #self.teams = self.get_teams()
        self.API_KEY = 'P03vXhW3gbgDaVjCG2Q/WUG/9wVH+HjLalL8S0z6m6fazd5wI904GdSqhwJUPYgx'
    def get_teams(self):
        df = pd.read_csv('teams.csv')
        return(df)
    teams = pd.read_csv('teams.csv')
    confs = pd.read_csv('confs.csv')
    def get_games_from_file(f='scrubbed_games_df2.csv'):
            df = pd.read_csv(f)
            #del df['home_line_scores']
            #del df['away_line_scores']
            del df['away_id']
            del df['notes']
            del df['start_time_tbd']
            del df['venue']
            del df['venue_id']  
            del df['away_post_win_prob']
            del df['away_postgame_elo']
            del df['excitement_index']
            del df['highlights']
            del df['home_post_win_prob']
            del df['home_postgame_elo']
            del df['home_pregame_elo']
            del df['id']
            del df['start_date']
            del df['away_pregame_elo']
            del df['home_id']
            df['attendance'].fillna(value=-1, inplace=True) 
            return(df)


    def get_games_from_cfbd(self,year):
        configuration = cfbd.Configuration()
        configuration.api_key['Authorization'] = 'P03vXhW3gbgDaVjCG2Q/WUG/9wVH+HjLalL8S0z6m6fazd5wI904GdSqhwJUPYgx'
        configuration.api_key_prefix['Authorization'] = 'Bearer'

        api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
        games = api_instance.get_games(year)

        #delete lists from the json
        tmp = games[0].to_dict()
        del tmp['home_line_scores']
        del tmp['away_line_scores']
        del tmp['away_id']
        del tmp['notes']
        del tmp['start_time_tbd']
        del tmp['venue']
        del tmp['venue_id']
        del tmp['away_post_win_prob']
        del tmp['away_postgame_elo']
        del tmp['excitement_index']
        del tmp['highlights']
        del tmp['home_post_win_prob']
        del tmp['home_postgame_elo']
        del tmp['home_pregame_elo']
        del tmp['id']
        del tmp['start_date']
        del tmp['away_pregame_elo']
        del tmp['home_id']
        #del tmp['venue_id']

        df = pd.DataFrame([tmp])

        for i in range(1,len(games)-1):
            tmp = games[i].to_dict()
            del tmp['home_line_scores']
            del tmp['away_line_scores']
            del tmp['away_id']
            del tmp['notes']
            del tmp['start_time_tbd']
            del tmp['venue']
            del tmp['venue_id']
            del tmp['away_post_win_prob']
            del tmp['away_postgame_elo']
            del tmp['excitement_index']
            del tmp['highlights']
            del tmp['home_post_win_prob']
            del tmp['home_postgame_elo']
            del tmp['home_pregame_elo']
            del tmp['id']
            del tmp['start_date']
            del tmp['away_pregame_elo']
            del tmp['home_id']
            df = df.append(pd.DataFrame([tmp]))
        df['attendance'].fillna(value=-1, inplace=True)    
        return(df) 

    def test():
        return(1)

    # the columns excluded from training. These are this week's game
    # level atrributes such as attendance for this game, this game's 
    # matchup, etc
    cols = ['team','opp','target', 'spread', 'team_conf_week_minus_one', 'team_conf_week_minus_two'
            , 'team_conf_week_minus_three', 'team_conf_week_minus_four', 'opp_conf_week_minus_one','opp_conf_week_minus_two'
           ,'opp_conf_week_minus_three','opp_conf_week_minus_four', 'team_conf_week_minus_five','opp_conf_week_minus_five']
    # model specific functions. Helpers for getting data, constructing
    # training sets
    def get_records(self,df,confs):
        records_inc = pd.DataFrame(columns=['team','opp','home_conference', 'home_conf_label','away_conference'
            ,'away_conf_label' ,'home_wins','home_losses','home_win', 'home_loss','spread'
            ,'away_wins','away_losses', 'week'
            , 'home_win_pct','total_wins','total','away_win_pct','win_pct' ])
        for wks in range(2,15,1):
            res = ps.sqldf("""
                    with tmp as (
                       select home_team, away_team
                       , case when home_points > away_points then 1 else 0 end as home_win
                       , case when home_points < away_points then 1 else 0 end as home_loss 
                       from df
                       where week < '""" + str(wks) + """'
                       order by home_team
                    ), home_df as
                    (
                        select home_team, sum(home_win) as home_wins
                        , sum(home_loss) as home_losses 
                        from tmp group by home_team
                    ) 
                    , tmp_by_week as (
                        select home_team, away_team, cast((home_points - away_points)as float) as spread
                        ,home_conference, away_conference
                       , case when home_points > away_points then 1 else 0 end as home_win
                       , case when home_points < away_points then 1 else 0 end as home_loss 
                       from df
                       where week = '""" + str(wks) + """'
                       --order by home_team
                    ) , tmp2 as
                    (
                        select away_team, case when home_points > away_points then 1 else 0 end as away_loss
                        , case when home_points < away_points then 1 else 0 end as away_win 
                        from df
                        where week < '""" + str(wks) + """'
                    ), away_df as 
                    (
                        select away_team, sum(away_win) as away_wins, sum(away_loss) as away_losses 
                        from tmp2 group by away_team
                    ), tots as ( 
                        select t.home_team team ,w.away_team opp, w.home_conference, c.conf_label home_conf_label
                        , w.away_conference, c2.conf_label away_conf_label
                        , home_wins,home_losses,home_win,home_loss, w.spread
                        from home_df t inner join tmp_by_week w
                        on t.home_team = w.home_team
                        inner join confs c on w.home_conference = c.conference
                        inner join confs c2 on w.away_conference = c2.conference
                    ), final as 
                    (
                        select t.*, a.away_wins,a.away_losses, '""" + str(wks) + """' as week
                        ,cast(home_wins as float) / (home_wins + home_losses) as home_win_pct
                        ,(home_wins + away_wins) as total_wins
                        ,((home_wins + away_wins) + home_losses + away_losses) as total
                        ,cast(away_wins as float) / (away_wins + away_losses) away_win_pct
                        ,cast((home_wins + away_wins) as float) / ((home_wins + away_wins) + home_losses + away_losses) as win_pct
                        from away_df a inner join tots t on a.away_team = t.team
                    ) 
                    select t.*
                    from final t --inner join records r 
                    --on t.team = r.team
                    """)
            records_inc = records_inc.append(res,sort=False)
        #records_inc
        opp_pct = ps.sqldf("""
            with tmp as (
                select home_team team , away_team opp, week
                from df 
                union all 
                select away_team team , home_team opp, week
                from df

            ) 
            , opp2 as(
                select o.opp team, o.week, win_pct opp_win_pct
                , home_win_pct opp_home_win_pct, away_win_pct opp_away_win_pct  
                from tmp o inner join records_inc r on o.opp = r.team and o.week = r.week
            )
            select * from opp2

            """    
        )


        records = ps.sqldf("""
            select r.*,  opp_win_pct
                ,  opp_home_win_pct,  opp_away_win_pct  
            from records_inc r inner join opp_pct o
            on r.team = o.team and r.week = o.week
            """)
        return(records)

    def get_training_data(self, wks, records, confs, teams):
        #wks = 7
        #teams = ps.sqldf("""
        #    with cte as( 
        #    select distinct  team 
        #    from records 
        ##    union all 
        #    select distinct opp team 
        #    from records
        #    )
        #    select distinct team from cte
        #    """)
        #teams['team_label'] = list(range(1, (len(teams.team)+1), 1))

        matchups = ps.sqldf("""     
        select  r.team, r.opp, t.team_label, t2.team_label opp_label, c.conference, c.conf_label team_conf_label
        , c2.conference, c2.conf_label opp_conf_label
        , home_win target, spread  
        from  records  r  inner join teams t on r.team = t.team 
        inner join teams t2 on r.opp = t2.team 
        inner join confs c on c.conference = r.home_conference
        inner join confs c2 on c2.conference = r.home_conference
        where week = '""" + str(wks) + """'

        """) 
    ########
    # week one
        week_minus_one_home = ps.sqldf("""     
        select  team,home_win_pct as home_win_pct_week_minus_one 
        ,away_win_pct as away_win_pct_week_minus_one
        ,win_pct as win_pct_week_minus_one
        ,spread spread_week_minus_one
        ,home_conference team_conf_week_minus_one,  home_conf_label team_conf_label_week_minus_one
        ,away_conference opp_conf_week_minus_one,  away_conf_label opp_conf_label_week_minus_one
        --,opp_win_pct,opp_home_win_pct,opp_away_win_pct 
        from  records  r   
        where week = '""" + str(wks-1) + """'
        and (team in (select team from matchups )   )
        """)
        week_minus_one_away = ps.sqldf(""" 
        select  opp ,opp_home_win_pct as opp_home_win_pct_week_minus_one
        ,opp_away_win_pct as opp_away_win_pct_week_minus_one
        ,opp_win_pct as opp_home_win_pct_one
        --,opp_win_pct,opp_home_win_pct,opp_away_win_pct 
        from  records  r   
        where week = '""" + str(wks-1) + """'
        and (opp in (select opp from matchups )   )
        """)
        ########
        # week two
        week_minus_two_home = ps.sqldf("""     
        select  team,home_win_pct as home_win_pct_week_minus_two
        ,away_win_pct as away_win_pct_week_minus_two
        ,win_pct as win_pct_week_minus_two
        ,spread spread_week_minus_two
        ,home_conference team_conf_week_minus_two,  home_conf_label team_conf_label_week_minus_two
        ,away_conference opp_conf_week_minus_two,  away_conf_label opp_conf_label_week_minus_two
        from  records  r   
        where week = '""" + str(wks-2) + """'
        and (team in (select team from matchups )   )
        """)
        week_minus_two_away = ps.sqldf(""" 
            select  opp ,opp_home_win_pct as opp_home_win_pct_week_minus_two
            ,opp_away_win_pct as opp_away_win_pct_week_minus_two
            ,opp_win_pct as opp_home_win_pct_two 
            from  records  r   
            where week = '""" + str(wks-2) + """'
            and (opp in (select opp from matchups )   )
        """)
        ########
        # week three
        week_minus_three_home = ps.sqldf("""     
            select  team,home_win_pct as home_win_pct_week_minus_three
            ,away_win_pct as away_win_pct_week_minus_three
            ,win_pct as win_pct_week_minus_three
            ,spread spread_week_minus_three
            ,home_conference team_conf_week_minus_three,  home_conf_label team_conf_label_week_minus_three
            ,away_conference opp_conf_week_minus_three,  away_conf_label opp_conf_label_week_minus_three
            from  records  r   
            where week = '""" + str(wks-3) + """'
            and (team in (select team from matchups )   )
            """)
        week_minus_three_away = ps.sqldf(""" 
            select  opp ,opp_home_win_pct as opp_home_win_pct_week_minus_three
            ,opp_away_win_pct as opp_away_win_pct_week_minus_three
            ,opp_win_pct as opp_home_win_pct_three
            from  records  r   
            where week = '""" + str(wks-3) + """'
            and (opp in (select opp from matchups )   )
        """)
        ########
        # week four
        week_minus_four_home = ps.sqldf("""     
            select  team,home_win_pct as home_win_pct_week_minus_four
            ,away_win_pct as away_win_pct_week_minus_four
            ,win_pct as win_pct_week_minus_four
            ,spread spread_week_minus_four
            ,home_conference team_conf_week_minus_four,  home_conf_label team_conf_label_week_minus_four
            ,away_conference opp_conf_week_minus_four,  away_conf_label opp_conf_label_week_minus_four
            from  records  r   
            where week = '""" + str(wks-4) + """'
            and (team in (select team from matchups )   )
            """)
        week_minus_four_away = ps.sqldf(""" 
            select  opp ,opp_home_win_pct as opp_home_win_pct_week_minus_four
            ,opp_away_win_pct as opp_away_win_pct_week_minus_four
            ,opp_win_pct as opp_home_win_pct_four
            from  records  r   
            where week = '""" + str(wks-4) + """'
            and (opp in (select opp from matchups )   )
        """)
        ########
        # week five
        week_minus_five_home = ps.sqldf("""     
            select  team,home_win_pct as home_win_pct_week_minus_five
            ,away_win_pct as away_win_pct_week_minus_five
            ,win_pct as win_pct_week_minus_five
            ,spread spread_week_minus_five
            ,home_conference team_conf_week_minus_five,  home_conf_label team_conf_label_week_minus_five
            ,away_conference opp_conf_week_minus_five,  away_conf_label opp_conf_label_week_minus_five
            from  records  r   
            where week = '""" + str(wks-5) + """'
            and (team in (select team from matchups )   )
            """)
        week_minus_five_away = ps.sqldf(""" 
            select  opp ,opp_home_win_pct as opp_home_win_pct_week_minus_five
            ,opp_away_win_pct as opp_away_win_pct_week_minus_five
            ,opp_win_pct as opp_home_win_pct_five
            from  records  r   
            where week = '""" + str(wks-5) + """'
            and (opp in (select opp from matchups )   )
        """)
        df_train = ps.sqldf("""  
            select m.team, m.opp, target, spread, m.team_label, m.opp_label
                ,coalesce(home_win_pct_week_minus_one,0) home_win_pct_week_minus_one
                ,coalesce(away_win_pct_week_minus_one,0) away_win_pct_week_minus_one
                ,coalesce(win_pct_week_minus_one ,0) win_pct_week_minus_one 
                ,coalesce(spread_week_minus_one ,0) spread_week_minus_one
                ,coalesce(team_conf_label_week_minus_one ,0) team_conf_label_week_minus_one
                ,team_conf_week_minus_one
                ,opp_conf_week_minus_one
                ,coalesce(opp_conf_label_week_minus_one ,0) away_conf_label_week_minus_one

                ,coalesce(opp_home_win_pct_week_minus_one,0) opp_home_win_pct_week_minus_one
                ,coalesce(opp_away_win_pct_week_minus_one,0) opp_away_win_pct_week_minus_one
                ,coalesce(opp_home_win_pct_one,0) opp_home_win_pct_one

                --
                ,coalesce(home_win_pct_week_minus_two,0) home_win_pct_week_minus_two
                ,coalesce(away_win_pct_week_minus_two,0) away_win_pct_week_minus_two
                ,coalesce(win_pct_week_minus_two ,0) win_pct_week_minus_two 
                ,coalesce(spread_week_minus_two ,0) spread_week_minus_two
                ,coalesce(team_conf_label_week_minus_two ,0) team_conf_label_week_minus_two
                ,team_conf_week_minus_two
                ,opp_conf_week_minus_two
                ,coalesce(opp_conf_label_week_minus_two ,0) away_conf_label_week_minus_two

                ,coalesce(opp_home_win_pct_week_minus_two,0) opp_home_win_pct_week_minus_two
                ,coalesce(opp_away_win_pct_week_minus_two,0) opp_away_win_pct_week_minus_two
                ,coalesce(opp_home_win_pct_two,0) opp_home_win_pct_two

                --
                ,coalesce(home_win_pct_week_minus_three,0) home_win_pct_week_minus_three
                ,coalesce(away_win_pct_week_minus_three,0) away_win_pct_week_minus_three
                ,coalesce(win_pct_week_minus_three ,0) win_pct_week_minus_three
                ,coalesce(spread_week_minus_three ,0) spread_week_minus_three
                ,coalesce(team_conf_label_week_minus_three ,0) team_conf_label_week_minus_three
                ,team_conf_week_minus_three
                ,opp_conf_week_minus_three
                ,coalesce(opp_conf_label_week_minus_three ,0) away_conf_label_week_minus_three

                ,coalesce(opp_home_win_pct_week_minus_three,0) opp_home_win_pct_week_minus_three
                ,coalesce(opp_away_win_pct_week_minus_three,0) opp_away_win_pct_week_minus_three
                ,coalesce(opp_home_win_pct_three,0) opp_home_win_pct_three
                --
                ,coalesce(home_win_pct_week_minus_four,0) home_win_pct_week_minus_four
                ,coalesce(away_win_pct_week_minus_four,0) away_win_pct_week_minus_four
                ,coalesce(win_pct_week_minus_four ,0) win_pct_week_minus_four
                ,coalesce(spread_week_minus_four ,0) spread_week_minus_four
                ,coalesce(team_conf_label_week_minus_four ,0) team_conf_label_week_minus_four
                ,team_conf_week_minus_four
                ,opp_conf_week_minus_four
                ,coalesce(opp_conf_label_week_minus_four ,0) away_conf_label_week_minus_four

                ,coalesce(opp_home_win_pct_week_minus_four,0) opp_home_win_pct_week_minus_four
                ,coalesce(opp_away_win_pct_week_minus_four,0) opp_away_win_pct_week_minus_four
                ,coalesce(opp_home_win_pct_four,0) opp_home_win_pct_four
                
                --
                ,coalesce(home_win_pct_week_minus_five,0) home_win_pct_week_minus_five
                ,coalesce(away_win_pct_week_minus_five,0) away_win_pct_week_minus_five
                ,coalesce(win_pct_week_minus_five ,0) win_pct_week_minus_five
                ,coalesce(spread_week_minus_five ,0) spread_week_minus_five
                ,coalesce(team_conf_label_week_minus_five ,0) team_conf_label_week_minus_five
                ,team_conf_week_minus_five
                ,opp_conf_week_minus_five
                ,coalesce(opp_conf_label_week_minus_five ,0) away_conf_label_week_minus_five

                ,coalesce(opp_home_win_pct_week_minus_five,0) opp_home_win_pct_week_minus_five
                ,coalesce(opp_away_win_pct_week_minus_five,0) opp_away_win_pct_week_minus_five
                ,coalesce(opp_home_win_pct_five,0) opp_home_win_pct_five
                from matchups m 
                left join week_minus_one_home oh
                on m.team = oh.team
                left join week_minus_one_away oa
                on m.opp = oa.opp
                --
                left join week_minus_two_home th
                on m.team = th.team
                left join week_minus_two_away ta
                on m.opp = ta.opp
                --
                left join week_minus_three_home thh
                on m.team = thh.team
                left join week_minus_three_away tha
                on m.opp = tha.opp
                --
                left join week_minus_four_home fh
                on m.team = fh.team
                left join week_minus_four_away fa
                on m.opp = fa.opp
                --
                left join week_minus_five_home fih
                on m.team = fih.team
                left join week_minus_five_away fia
                on m.opp = fia.opp
            """)

        return(df_train)
    def get_wins_losses_from_week(self):
        return(0)
    def get_score(self, df_train):
        X = df_train.loc[:, ~df_train.columns.isin(cols)]
        y = df_train.loc[: ,df_train.columns=='target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        clf.fit(X_train, y_train.values.ravel())
        df_train['win_prob'] = clf.predict_proba(df_train.loc[:, ~df_train.columns.isin( cols)])[:,1]
        #df_train['pred'] = clf.predict(df_train.loc[:, ~df_train.columns.isin( cols)])
        score = clf.score(X_test, y_test)
        return(score)


    ######################################################
    # this currently does not work. Complains about the fact that I pass a string and it wants a list
    ######################################################
    def get_results(df_train, tm):
        t = list(tm)
        if(df_train.team.isin(t)):
            return(df_train[['team','opp','win_prob']][df_train.team == str(t)])
        else:
            return(df_train[['team','opp','win_prob']][df_train.opp == str(t)]) 