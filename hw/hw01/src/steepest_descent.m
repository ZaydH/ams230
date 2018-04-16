function steepest_descent(f, grad, d_phi, c, num_iteration, c1, c2)

    x = zeros(num_iterations, 1);
    err = zeros(num_iterations, 1);
    for i = 1:(num_iterations)
        p_k = grad(x[i])
        
        x(i), err(i) = line_search_itr(xk, c1,c2)
    end

    plot_line_search_results(x, err);
end


function err = line_search_itr(grad, xk, x_opt c1,c2)
    pk = -grad(xk);
    alpha = line_search(phi, d_phi, c1, c2, 1000);
    
    x_k1 = xk + alpha * pk;
    err = abs(xk - x_opt);
end


function plot_line_search_results(x, err, c)
    iter_count = 1:length(x);
    
    % Generate the plot of cost function f(x)
    figure
    subplot(2, 1, 1);
    plot(iter_count, x);
    xlim([0 max(x)]);
    xlabel('Iteration #');
    ylabel('x_k');
    title({['Steepest Descent x for c=' int2str(c)]});
    set(gca,'YScale','log') % Log scale
    
    % Generate the plot of cost function f(x)
    figure
    subplot(2, 1, 2);
    plot(x, err)
    xlim([0 max(x)]);
    xlabel('Iteration #');
    ylabel('Error');
    title({['Steepest Descent Error for c=' int2str(c)]});
    set(gca,'YScale','log') % Log scale
end