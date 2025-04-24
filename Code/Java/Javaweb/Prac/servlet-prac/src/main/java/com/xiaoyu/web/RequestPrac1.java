package com.xiaoyu.web;

import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.WebServlet;
import java.io.IOException;

@WebServlet("/request1")
public class RequestPrac1 extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String method = request.getMethod();
        System.out.println("Request method: " + method);

        String contextPath = request.getContextPath();
        System.out.println("Context path: " + contextPath);

        StringBuffer requestURL = request.getRequestURL();
        System.out.println("Request URL: " + requestURL.toString());

        String requestURI = request.getRequestURI();
        System.out.println("Request URI: " + requestURI);

        String queryString = request.getQueryString();
        System.out.println("Query string: " + queryString);

        // User-Agent is version information of the browser
        String agent = request.getHeader("User-Agent");
        System.out.println("User-Agent: " + agent);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        BufferedReader reader = request.getReader();
        String line = reader.readLine();
        System.out.println("Request body: " + line);
    }
}
