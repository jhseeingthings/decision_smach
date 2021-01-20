function result = rnn_game_result(tmp_output)

tmp_payoff = zeros(12,size(tmp_output,2));
tmp_payoff(1:6,:) = tmp_output(1:6,:);
tmp_payoff(7:10,:) = tmp_output(1:4,:);
tmp_payoff(11:12,:) = tmp_output(7:8,:);

tmp_reshape = reshape(tmp_payoff(:,1),[],2);
[action] = npg([3,2],tmp_reshape);
tmp_action = find(action(:,1) == max(action(:,1)));
tmp_action2 = find(action(:,2) == max(action(:,2)));

switch tmp_action
    case 1
        result = 1;
    case 2
        result = 2;
    case 3
        result = 3;
end