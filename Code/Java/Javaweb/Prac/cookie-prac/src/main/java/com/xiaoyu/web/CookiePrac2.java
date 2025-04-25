package com.xiaoyu.web;

import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.WebServlet;
import java.io.IOException;

@WebServlet("/cookie2")
public class CookiePrac2 extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        Cookie[] cookies = request.getCookies();

        for (Cookie cookie : cookies) {
            String name = cookie.getName();
            if ("username".equals(name)){
                String value = cookie.getValue();
                System.out.println(name+" = " + value);
            }
        }
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doGet(request, response);
    }
}
