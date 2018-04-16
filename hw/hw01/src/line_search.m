function alpha = line_search(phi, d_phi, c1, c2, max_loop_count)
    % Store the default values
    phi_0 = phi(0);
    d_phi_0 = d_phi(0);
    
    % "alpha" is set via a doubling routine
    alpha_prev = 0;
    phi_alpha = phi(alpha_prev);
    alpha = 1;
    for i = 1:max_loop_count
        phi_alpha_prev = phi_alpha;
        phi_alpha = phi(alpha);
        if (phi_alpha > phi_0 + c1 * alpha * d_phi_0 ...
                || (phi_alpha >= phi_alpha_prev && i > 1))
            alpha = line_search_zoom(phi, alpha_prev, alpha);
            return
        end
        
        d_phi_alpha = d_phi(alpha);
        if (abs(d_phi_alpha) <= -c2 * d_phi_0)
            % No need to update alpha
            return
        end
        if (d_phi_alpha >= 0)
            alpha = line_search_zoom(phi, alpha, alpha_prev);
            return
        end
        
        % Increase the alpha in a doubling step.
        alpha_prev = alpha;
        alpha = alpha * 2;
    end
end


function alpha = line_search_zoom(alpha_lo, alpha_hi, phi, phi_0, ...
                                  d_phi, d_phi_0, c1, c2)
    while(true)
        alpha_m = (alpha_lo + alpha_hi)/2;  % Bisection
        phi_alpha = phi(alpha_m);
        if (phi_alpha > phi_0 + c1 * alpha_m*d_phi_0 ...
                || phi_alpha >= phi(alpha_lo))
            alpha_hi = alpha_m;
        else
            d_phi_alpha = d_phi(alpha_m);
            if (abs(d_phi_alpha) <= -c2 * d_phi_0)
                alpha = alpha_m;
                return;
            end
            if (d_phi_alpha*(alpha_hi - alpha_lo) >= 0)
                alpha_hi = alpha_lo;
            end
            alpha_lo = alpha_m;
        end
    end
end