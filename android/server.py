from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf


def main() -> None:
    # Create strategy
    strategy = fl.server.strategy.FedAvgAndroid(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_fn=None,
        on_fit_config_fn=fit_config,
        initial_parameters=None,
    )

    # Start Flower server for 10 rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 5,
    }
    return config


if __name__ == "__main__":
    main()



public InfoViewModel()
        {
            Task.Run(async () => await GetUserData());
            //async void Execute() => await GoToLeaderboard();
            //ClickedCommand = new Command(Execute);
        }

        private async Task GoToLeaderboard()
        {
            //await Shell.Current.GoToAsync($"//{nameof(LeaderBoardPage)}");
        }
        public async Task GetUserData()
        {
            IsBusy = true;
            var userTask = RequestUtils.GetUserData();
            //var rankTask = RequestUtils.GetLeaderboard();
            await Task.WhenAll(userTask);
            
            var objData = userTask.Result;
            //var objRank = rankTask.Result;
            /*
            var ranks = (JArray) objRank["leaderboard"];

            if (ranks != null)
            {
                for (var i = 0; i < ranks.Count; i++)
                {
                    ranks = new JArray(ranks.OrderByDescending(x => x["value"]));
                    if (RestFiwareUtils.GetUserId() == ranks[i]["student_id"]?.ToString())
                    {
                        Results = ranks[i]["value"]?.ToString();
                        IndResults = (i + 1).ToString();

                        NumColor = Color.FromHex(IndResults == "1" ? "#FFD700" : "#36D7B7");

                        Ordinal = (i + 1) switch
                        {
                            1 => "st",
                            2 => "nd",
                            3 => "rd",
                            _ => "th"
                        };
                        
                    }
                }
            }*/
			
			
			 FiwareService = "maori";