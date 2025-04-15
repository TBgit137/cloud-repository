package com.xiaoyu.jdbc;

import com.alibaba.druid.pool.DruidDataSource;
import com.alibaba.druid.pool.DruidDataSourceFactory;

import javax.sql.DataSource;
import java.io.FileInputStream;
import java.sql.Connection;
import java.util.Properties;

public class DruidPrac {
    public static void main(String[] args) throws Exception {
        // import jar package


        Properties prop = new Properties();
        prop.load(new FileInputStream("JDBC/jdbc-prac/src/druid.properties"));

        DataSource dataSource = DruidDataSourceFactory.createDataSource(prop);

        Connection connection = dataSource.getConnection();
    }
}
