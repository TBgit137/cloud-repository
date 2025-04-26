package com.xiaoyu.web.filter;

import javax.servlet.*;
import javax.servlet.annotation.WebFilter;
import java.io.IOException;

@WebFilter("/*")
public class FilterPrac1 implements Filter {
    @Override
    public void init(FilterConfig filterConfig) throws ServletException {

    }

    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain) throws IOException, ServletException {
        System.out.println("Filter1 is working");

        filterChain.doFilter(servletRequest, servletResponse);

        System.out.println("Filter1 after doFilter logic");
    }

    @Override
    public void destroy() {

    }
}
