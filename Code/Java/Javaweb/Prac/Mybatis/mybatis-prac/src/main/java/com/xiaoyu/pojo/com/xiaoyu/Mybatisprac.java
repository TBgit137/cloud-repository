package com.xiaoyu.pojo.com.xiaoyu;

import com.xiaoyu.pojo.User;
import org.apache.ibatis.io.Resources;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.session.SqlSessionFactoryBuilder;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class Mybatisprac {
    public static void main(String[] args) throws IOException {

        // Load the MyBatis configuration file
        String resource = "mybatis-config.xml";
        InputStream inputStream = Resources.getResourceAsStream(resource);
        SqlSessionFactory sqlSessionFactory = new SqlSessionFactoryBuilder().build(inputStream);

        // Get a SqlSession object to execute SQL statements
        SqlSession sqlSession = sqlSessionFactory.openSession();

        // Execute a SQL statement
        List<User> users = sqlSession.selectList("test.selectAll");

        System.out.println(users);

        // Close the SqlSession
        sqlSession.close();
    }
}
