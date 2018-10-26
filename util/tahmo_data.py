import json
def merge_tahmo_chirp_csv(chirp_file, tahmo_file):
    """
    Merge both tahmo and chirp data. This assumes there are similar stations name
    in both chirp and tahmo station.
    @param: chirp_file json file for chirp data.
    @param: tahmo_file json file for tahmo data.
    """
    from pandas import DataFrame
    js_tahmo = json.load(open(tahmo_file,'r'))
    js_chirp = json.load(open(chirp_file,'r'))
    merged_data = []
    for station, st_value in js_tahmo.items():
        for yr,yr_value in st_value.items():
            for mnt, mnt_value in yr_value.items():
                for pentd, value in mnt_value.items():
                    chirp_value = js_chirp[station][yr][mnt][pentd]
                    row = [station] + [yr] + [mnt] + [pentd] + [value] + [chirp_value]
                    merged_data.append(row)
    pn_tahmo = DataFrame(merged_data)
    pn_tahmo.rename(columns={0:'station',1:'year',2:'month',3:'pentday',4:'tahmo',5:'chirp'},inplace=True)
    pn_tahmo.sort_values(by=['station','year','month','pentday'],inplace=True)
    pn_tahmo.to_csv('chirp_tahmo.csv')


