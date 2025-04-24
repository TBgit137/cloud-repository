package com.xiaoyu.web;

import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.WebServlet;
import java.io.IOException;
import java.util.Map;

@WebServlet("/request2")
public class RequestPrac2 extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        System.out.println("get method running");

        Map<String, String[]> map = request.getParameterMap();
        for (String key : map.keySet()) {
            System.out.print(key+":");

            String[] vals = map.get(key);
            for (String val : vals) {
                System.out.print(val+" ");
            }
            System.out.println();
        }

        System.out.println("-----------------");
        String[] hobbies = request.getParameterValues("hobby");
        for (String hobby : hobbies) {
            System.out.println(hobby);
        }
        System.out.println("-----------------");
        String username = request.getParameter("username");
        String password = request.getParameter("password");
        System.out.println(username);
        System.out.println(password);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    }
}
